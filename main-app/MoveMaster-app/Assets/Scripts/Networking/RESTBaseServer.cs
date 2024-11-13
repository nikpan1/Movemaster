using UnityEngine;
using System.Net.Http;
using System.Threading.Tasks;
using System;
using System.Net;
using System.IO;
using System.Text;
using System.Collections.Concurrent;
using System.Threading;


public class RESTBaseServer : MonoBehaviour, IDisposable
{
    #region Endpoint storage
    private delegate string EndpointFunction(string input);
    private ConcurrentDictionary<RESTEndpoint, EndpointFunction> endpointActions = new();
    #endregion

    #region Server settings
    private bool isListenerRunning = false;
    private readonly string serverUrl = "http://localhost:7000/";
    #endregion

    private readonly HttpClient httpClient = new();
    private readonly HttpListener httpListener = new();

    #region Singleton Implementation
    private static RESTBaseServer _instance;

    public static RESTBaseServer Instance
    {
        get
        {
            if (_instance == null)
            {
                _instance = FindObjectOfType<RESTBaseServer>();
                if (_instance == null)
                {
                    GameObject singletonObject = new GameObject("RESTBaseServer");
                    _instance = singletonObject.AddComponent<RESTBaseServer>();
                }
            }
            return _instance;
        }
    }

    private void Start()
    {
        if (_instance != null && _instance != this)
        {
            Destroy(gameObject);
            return;
        }

        _instance = this;

        // Making the singleton persist across scenes
        DontDestroyOnLoad(gameObject);
    }
    #endregion

    private void Log(string message) => Debug.Log("[REST Base Server] : " + message);
    private void LogError(string message) => Debug.LogError("[REST Base Server] : " + message);
    private void Awake() => StartListener();
    private void OnDestroy() => StopListener();


    private void StartListener()
    {
        RegisterAction(ApiMethod.GET, "/healthcheck", HandleHealthCheck);

        isListenerRunning = true;

        if (!HttpListener.IsSupported)
        {
            LogError("HttpListener is not supported on this platform.");
            return;
        }

        httpListener.Prefixes.Add(serverUrl);
        httpClient.Timeout = TimeSpan.FromSeconds(30);
        httpListener.Start();

        Task.Run(() => ListenForRequests());
    }

    private void StopListener()
    {
        isListenerRunning = false;
        httpListener.Stop();
    }

    private async Task ListenForRequests()
    {
        var taskQueue = new ConcurrentQueue<Task>();

        while (isListenerRunning && httpListener.IsListening)
        {
            try
            {
                HttpListenerContext context = await httpListener.GetContextAsync();
                taskQueue.Enqueue(HandleRequestAsync(context));
            }
            catch (ObjectDisposedException) when (!isListenerRunning)
            {
                Log($"Listener has been closed gracefully.");
            }

            // Process queued requests concurrently.
            while (taskQueue.TryDequeue(out var task))
            {
                await task;  // Handle requests asynchronously in parallel
            }
            await Task.Delay(100);
        }
    }
    private async Task HandleRequestAsync(HttpListenerContext context)
    {
        try
        {
            HttpListenerRequest request = context.Request;
            string requestContent = await ReadRequestContent(request);

            RESTEndpoint endpointType = new(request.Url.AbsolutePath, request.HttpMethod.ToApiMethod());
            string responseContent = CallEndpoint(endpointType, requestContent);

            await SendResponse(context.Response, responseContent);
        }
        catch (Exception ex)
        {
            LogError($"Error handling request: {ex.Message}");
        }
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

    private static SemaphoreSlim semaphore = new SemaphoreSlim(10); // Limit to 10 concurrent requests

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

    public string CallEndpoint(RESTEndpoint endpointType, string input)
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