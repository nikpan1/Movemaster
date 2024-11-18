using UnityEngine;
using System.Net.Http;
using System.Threading.Tasks;
using System;
using System.Net;
using System.IO;
using System.Text;
using System.Collections.Concurrent;
using System.Threading;


public class RestBaseServer : IDisposable
{
    #region Endpoint Management
    private delegate string EndpointFunction(string input);
    private ConcurrentDictionary<RESTEndpoint, EndpointFunction> _endpointActions = new();

    private class EndpointQueue
    {
        public ConcurrentQueue<HttpListenerContext> Queue = new();
        public bool IsProcessing = false;
    }

    private readonly ConcurrentDictionary<string, EndpointQueue> _endpointQueues = new();
    #endregion

    #region Server Configuration
    private const string ServerUrl = "http://localhost:7000/";

    private readonly HttpClient _client = new();
    private readonly HttpListener _listener = new();
    private static readonly SemaphoreSlim RequestConcurrencySemaphore = new(8); // renamed for clarity

    private bool _isListenerRunning = false;
    #endregion

    #region Logging Utilities

    private void Log(string message) => Debug.Log("[REST Base Server] : " + message);
    private void LogError(string message) => Debug.LogError("[REST Base Server] : " + message);
    #endregion
    
    
    public RestBaseServer()
    {
        if (!HttpListener.IsSupported)
        {
            throw new PlatformNotSupportedException("This platform does not support HTTP listeners.");
        }

        RegisterAction("/healthcheck", HttpMethod.Get, HandleHealthCheck);
        
        _listener.Prefixes.Add(ServerUrl);
        _client.Timeout = TimeSpan.FromSeconds(10);
        _listener.Start(); 
        
        _isListenerRunning = true;
    }
    
    public void StartListener()
    {
        Task.Run(ListenForRequests);
    }

    public void StopListener()
    {
        _isListenerRunning = false;
        _listener.Stop();
    }

    private async Task ListenForRequests()
    {
        while (_isListenerRunning && _listener.IsListening)
        {
            try
            {
                HttpListenerContext context = await _listener.GetContextAsync();
                string endpoint = context.Request.Url.AbsolutePath;
                var queue = _endpointQueues.GetOrAdd(endpoint, _ => new EndpointQueue());

                lock (queue)
                {
                    // Clear the queue and add the new request
                    while (queue.Queue.TryDequeue(out _));
                    queue.Queue.Enqueue(context);

                    if (!queue.IsProcessing)
                    {
                        queue.IsProcessing = true;
                        Task.Run(() => ProcessQueue(endpoint));
                    }
                }
            }
            catch (ObjectDisposedException) when (!_isListenerRunning)
            {
                Log($"Listener has been closed gracefully.");
            }
        }
    }

    private async Task ProcessQueue(string endpoint)
    {
        if (!_endpointQueues.TryGetValue(endpoint, out var queue))
            return;

        while (queue.Queue.TryDequeue(out var context))
        {
            try
            {
                await HandleRequest(context);
            }
            catch (Exception ex)
            {
                LogError($"Error processing request for endpoint {endpoint}: {ex.Message}");
                // Indicating an internal error to the requester
                context.Response.StatusCode = 500;
                context.Response.Close();
            }
        }

        lock (queue)
        {
            queue.IsProcessing = false;
        }
    }

    
    private async Task HandleRequest(HttpListenerContext context)
    {
        HttpListenerRequest request = context.Request;
        string requestContent = await ReadRequestContent(request);
        RESTEndpoint endpointType = new(request.Url.AbsolutePath, request.HttpMethod);
        
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
        await RequestConcurrencySemaphore.WaitAsync();

        try
        {
            HttpRequestMessage requestMessage = new();
            requestMessage.RequestUri = new Uri(url + endpoint.Url);
            requestMessage.Method = endpoint.Method;

            if (IsRequestTypeSupportingBody(endpoint, content))
            {
                requestMessage.Content = new StringContent(content, Encoding.UTF8, "application/json");
            }

            HttpResponseMessage response = await _client.SendAsync(requestMessage).ConfigureAwait(false);
            return await response.Content.ReadAsStringAsync().ConfigureAwait(false);
        }
        catch (ObjectDisposedException) when (!_isListenerRunning)
        {
            return null; //Listener has been closed gracefully. All good
        }
        catch (Exception ex)
        {
            LogError($"Error sending request to {url}: {ex.Message}");
            return null;
        }
        finally
        {
            RequestConcurrencySemaphore.Release();
        }
    }

    private bool IsRequestTypeSupportingBody(RESTEndpoint endpoint, string content)
    {
        return endpoint.Method != HttpMethod.Get && !string.IsNullOrEmpty(content);
    }

    public void RegisterAction(string endpointUrl, HttpMethod method, Func<string, string> func)
    {
        Log($"Endpoint added: {endpointUrl} ({method})"); 
        
        _endpointActions.AddOrUpdate(
            new RESTEndpoint(endpointUrl, method),
            _ => new EndpointFunction(func),
            (_, existing) => existing + new EndpointFunction(func)
        );
    }

    private string CallEndpoint(RESTEndpoint endpointType, string input)
    {
        Log($"Endpoint called: {endpointType.Url} ({endpointType.Method})");

        if (!_endpointActions.TryGetValue(endpointType, out var endpointDelegates))
        {
            throw new Exception($"Endpoint was not initialized: {endpointType.Url} ({endpointType.Method})");
        }

        string response = null;
        foreach (EndpointFunction endpoint in endpointDelegates.GetInvocationList())
        {
            response = endpoint(input) ?? response; // if the endpoint call returns a non-null value, then store it as response, but don't override it
        }

        return response;
    }

    #region Dispose pattern implementation
    protected virtual void Dispose(bool disposing)
    {
        if (disposing)
        {
            _listener?.Close();
            _client?.Dispose();
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
        return JsonUtility.ToJson(new { status = "OK" });
    }
}