using UnityEngine;
using System.Net.Http;
using System.Threading.Tasks;
using System;
using System.Net;
using System.IO;
using System.Text;
using System.Collections.Concurrent;
using System.Threading;


public class RESTBaseServer : IDisposable
{
    #region Endpoint storage
    private delegate string EndpointFunction(string input);
    private ConcurrentDictionary<RESTEndpoint, EndpointFunction> endpointActions = new();
    #endregion

    #region Server settings
    private bool isListenerRunning = false;
    private readonly string serverUrl = "http://localhost:7000/";
    #endregion

    private class EndpointQueue
    {
        public ConcurrentQueue<HttpListenerContext> Queue = new();
        public bool IsProcessing = false;
    }

    private readonly ConcurrentDictionary<string, EndpointQueue> endpointQueues = new();
    
    private readonly HttpClient httpClient = new();
    private readonly HttpListener httpListener = new();

    private static SemaphoreSlim semaphore = new(8);

    // ReSharper disable Unity.PerformanceAnalysis
    private void Log(string message) => Debug.Log("[REST Base Server] : " + message);
    // ReSharper disable Unity.PerformanceAnalysis
    private void LogError(string message) => Debug.LogError("[REST Base Server] : " + message);
    
    public RESTBaseServer()
    {
        if (!HttpListener.IsSupported)
        {
            throw new PlatformNotSupportedException("This platform does not support HTTP listeners.");
        }

        RegisterAction(ApiMethod.GET, "/healthcheck", HandleHealthCheck);
        
        httpListener.Prefixes.Add(serverUrl);
        httpClient.Timeout = TimeSpan.FromSeconds(30);
        httpListener.Start(); 
        
        isListenerRunning = true;
    }
    
    public void StartListener()
    {
        Task.Run(ListenForRequests);
    }

    public void StopListener()
    {
        isListenerRunning = false;
        httpListener.Stop();
    }

    private async Task ListenForRequests()
    {
        while (isListenerRunning && httpListener.IsListening)
        {
            try
            {
                HttpListenerContext context = await httpListener.GetContextAsync();
                string endpoint = context.Request.Url.AbsolutePath;

                // Ensure an endpoint queue exists
                var queue = endpointQueues.GetOrAdd(endpoint, _ => new EndpointQueue());

                lock (queue)
                {
                    // Clear the queue and add the new request
                    while (queue.Queue.TryDequeue(out _)) { }
                    queue.Queue.Enqueue(context);

                    // If not already processing, start processing
                    if (!queue.IsProcessing)
                    {
                        queue.IsProcessing = true;
                        Task.Run(() => ProcessQueue(endpoint));
                    }
                }
            }
            catch (ObjectDisposedException) when (!isListenerRunning)
            {
                Log($"Listener has been closed gracefully.");
            }
            catch (Exception ex)
            {
                LogError($"Error in listener: {ex.Message}");
            }
        }
    }

    private async Task ProcessQueue(string endpoint)
    {
        if (!endpointQueues.TryGetValue(endpoint, out var queue))
            return;

        while (queue.Queue.TryDequeue(out HttpListenerContext context))
        {
            try
            {
                await HandleRequest(context);
            }
            catch (Exception ex)
            {
                LogError($"Error processing request for endpoint {endpoint}: {ex.Message}");
                context.Response.StatusCode = 500;
                context.Response.Close();
            }
        }

        // Mark the endpoint as no longer processing
        lock (queue)
        {
            queue.IsProcessing = false;
        }
    }

    
    private async Task HandleRequest(HttpListenerContext context)
    {
        HttpListenerRequest request = context.Request;

        string requestContent = await ReadRequestContent(request);

        RESTEndpoint endpointType = new(request.Url.AbsolutePath, request.HttpMethod.ToApiMethod());
        string responseContent = CallEndpoint(endpointType, requestContent);

        await SendResponse(context.Response, responseContent);
    }

    private async Task<string> ReadRequestContent(HttpListenerRequest request)
    {
        using StreamReader reader = new(request.InputStream, request.ContentEncoding);
        return await reader.ReadToEndAsync();
    }

    private async Task SendResponse(HttpListenerResponse response, string responseString)
    {
        byte[] buffer = Encoding.UTF8.GetBytes(responseString);
        response.ContentLength64 = buffer.Length;

        using Stream outputStream = response.OutputStream;
        await outputStream.WriteAsync(buffer, 0, buffer.Length);
    }
    
    public async Task<string> SendRequest(RESTEndpoint endpoint, string url, string content = null)
    {
        await semaphore.WaitAsync();  // Limit concurrency

        try
        {
            Log($"Request Send to {url} : {endpoint.Url} ({endpoint.Method})");
            if (!string.IsNullOrEmpty(content))
            {
                Log($"Request Content: {content}");
            }

            string fullUrl = url + endpoint.Url;

            HttpRequestMessage requestMessage = new()
            {
                RequestUri = new Uri(fullUrl),
                Method = endpoint.Method.ToHttpMethod(),
            };

            // Only set content if the method supports a body
            if (endpoint.Method != ApiMethod.GET && !string.IsNullOrEmpty(content))
            {
                requestMessage.Content = new StringContent(content, Encoding.UTF8, "application/json");
            }

            HttpResponseMessage response = await httpClient.SendAsync(requestMessage).ConfigureAwait(false);
            response.EnsureSuccessStatusCode();
            string responseContent = await response.Content.ReadAsStringAsync().ConfigureAwait(false);
            return responseContent;
        }
        catch (ObjectDisposedException) when (!isListenerRunning)
        {
            Log($"Listener has been closed gracefully.");
            return null;
        }
        catch (Exception ex)
        {
            LogError($"Error sending request to {url}: {ex.Message}");
            return null;
        }
        finally
        {
            semaphore.Release();  // Release the semaphore after the task completes
        }
    }

    public void RegisterAction(ApiMethod method, string url, Func<string, string> func)
    {
        RESTEndpoint endpointType = new(url, method);
        EndpointFunction endpointFunc = new(func);
        Log($"Endpoint added: {endpointType.Url} ({endpointType.Method})");

        if (!endpointActions.ContainsKey(endpointType))
        {
            endpointActions[endpointType] = endpointFunc;
        }
        else
        {
            endpointActions[endpointType] += endpointFunc;
        }
    }

    private string CallEndpoint(RESTEndpoint endpointType, string input)
    {
        Log("Endpoint called: " + endpointType.Url + " (" + endpointType.Method + ")");
        if (!endpointActions.ContainsKey(endpointType))
        {
            LogError($"Endpoint not found: {endpointType.Url} ({endpointType.Method})");
            throw new Exception("Endpoint not found");
        }

        string response = "";
        foreach (var endpoint in endpointActions[endpointType].GetInvocationList())
        {
            response = endpoint.DynamicInvoke(input) as string;
        }

        return response;
    }

    #region Dispose pattern implementation
    protected virtual void Dispose(bool disposing)
    {
        if (disposing)
        {
            httpListener?.Close();
            httpClient?.Dispose();
        }
    }

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }
    
    #endregion

    private string HandleHealthCheck(string input)
    {
        return "{\"status\": \"OK\"}";
    }
}