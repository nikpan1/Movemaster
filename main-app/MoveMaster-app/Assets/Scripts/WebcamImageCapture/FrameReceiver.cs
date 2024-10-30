using System;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;


public class FrameReceiver : MonoBehaviour
{
    private HttpListener httpListener;
    private static readonly HttpClient client = new HttpClient();
    private Thread listenerThread;

    private Sprite latestFrame;
    private string base64Frame;
    public Sprite LatestFrame => latestFrame;
    public string Base64Frame => base64Frame;

    private bool isRunning = false;
    private bool isCCSRunning = false;

    private const string shutdownEndpoint = "/shutdown/";
    private const string unityServerURL = "http://localhost:7000";
    private const int maxAttempts = 10;
    private const float retryDelay = 3.0f;

    private const string pShutdownEndpoint = "/shutdown";
    private const string healthCheckEndpoint = "/health";
    private const string ccCaptureEndpoint = "/capture_and_send";
    private const string captureCameraServerURL = "http://localhost:8001";
    private const string ccName = "Capture Camera Server";

    private async void Start()
    {
        HttpListenerSetup();
        isRunning = true;
        _ = StartListener();
        isCCSRunning = await CheckServerStatus(captureCameraServerURL, healthCheckEndpoint, ccName);

        if (isCCSRunning)
        {
            while (isRunning)
            {
                await StartImageCapture();
            }
        }
        else
        {
            Debug.LogError($"{ccName}: Server is not running. Unable to start continuous capture.");
        }
    }
    
    private async void OnApplicationQuit()
    {
        if (isCCSRunning)
        {
            await SendShutdownSignal();
        }
        StopListener();
    }

    private void HttpListenerSetup()
    {
        httpListener = new HttpListener();
        httpListener.Prefixes.Add(unityServerURL + shutdownEndpoint);
    }

    private async Task<bool> CheckServerStatus(string serverUrl, string healthEndpoint, string serverName)
    {
        for (int attempt = 0; attempt < maxAttempts; attempt++)
        {
            try
            {
                HttpResponseMessage statusResponse = await client.GetAsync(serverUrl + healthEndpoint);
                if (statusResponse.IsSuccessStatusCode)
                {
                    Debug.Log($"{serverName}: Server is running.");
                    return true;
                }
            }
            catch (Exception error)
            {
                Debug.LogError($"{serverName}: Check failed - {error.Message}");
            }
            await Task.Delay(TimeSpan.FromSeconds(retryDelay));
        }
        return false;
    }

    private async Task StartListener()
    {
        httpListener.Start();
        try
        {
            while (isRunning && httpListener.IsListening)
            {
                HttpListenerContext context = await httpListener.GetContextAsync();
                ProcessRequest(context);
            }
        }
        catch (HttpListenerException ex) when (ex.ErrorCode == 995)
        {
            Debug.Log("Listener has been closed gracefully.");
        }
        finally
        {
            httpListener.Close();
        }
    }

    private void StopListener()
    {
        isRunning = false;
        if (httpListener != null && httpListener.IsListening)
        {
            httpListener.Stop();
        }
    }

    private void ProcessRequest(HttpListenerContext context)
    {
        HttpListenerRequest request = context.Request;
        HttpListenerResponse response = context.Response;
        if (request.Url.AbsolutePath == pShutdownEndpoint)
        {
            ClientShutdownSignal(response, ccName);
        }
        response.Close();
    }

    private async Task SendShutdownSignal()
    {
        using (StringContent content = new StringContent("Unity server is shutting down", Encoding.UTF8, "application/json"))
        {
            HttpResponseMessage shutdownResponse = await client.PostAsync(captureCameraServerURL + pShutdownEndpoint, content);
            shutdownResponse.EnsureSuccessStatusCode();
        }
    }

    private void ClientShutdownSignal(HttpListenerResponse response, string serverName)
    {
        Debug.Log($"{serverName}: Shutdown signal received.");
        isCCSRunning = false;
        StopListener();
        response.StatusCode = (int)HttpStatusCode.OK;
        response.OutputStream.Close();
        Debug.Log($"{serverName}: Server is shutting down...");
    }

    private async Task StartImageCapture()
    {
        var request = (HttpWebRequest)WebRequest.Create(captureCameraServerURL + ccCaptureEndpoint);
        request.Method = "POST";

        try
        {
            using (var response = (HttpWebResponse)await request.GetResponseAsync())
            {
                if (response.StatusCode == HttpStatusCode.OK)
                {
                    using (var streamReader = new StreamReader(response.GetResponseStream()))
                    {
                        base64Frame = await streamReader.ReadToEndAsync();
                        base64Frame = base64Frame.Substring(1, base64Frame.Length - 2);
                        Debug.Log($"{ccName}: Capture initiated successfully and image received.");

                        byte[] imageBytes = Convert.FromBase64String(base64Frame);
                        Texture2D texture = new Texture2D(2, 2);
                        texture.LoadImage(imageBytes);
                        latestFrame = Sprite.Create(texture, new Rect(0, 0, texture.width, texture.height), new Vector2(0.5f, 0.5f));
                        Debug.Log($"Base64Frame content: {base64Frame.Substring(0, Mathf.Min(50, base64Frame.Length))}...");

                        if (latestFrame != null)
                        {
                            Debug.Log($"LatestFrame resolution: {latestFrame.texture.width}x{latestFrame.texture.height}");
                        }
                        else
                        {
                            Debug.Log("LatestFrame is null.");
                        }
                    }
                }
                else
                {
                    Debug.LogError($"{ccName}: Failed to start capture. Status code - {response.StatusCode}");
                }
            }
        }
        catch (Exception error)
        {
            Debug.LogError($"{ccName}: Error starting image capture - {error.Message}");
        }
    }

}