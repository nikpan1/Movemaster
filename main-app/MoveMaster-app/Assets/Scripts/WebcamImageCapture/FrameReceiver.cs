using Palmmedia.ReportGenerator.Core.CodeAnalysis;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.UIElements;
using CVServer.Models;

public class FrameReceiver : MonoBehaviour
{
    private HttpListener httpListener;
    private Thread listenerThread;
    private bool isRunning = false;
    private bool isCvAndCcRunning = false;
    private JointPosition[] joints = new JointPosition[32];

    private const int maxAttempts = 10;
    private static readonly HttpClient client = new HttpClient();
    private const string shutdownEndpoint = "/shutdown/";
    private const string unityServerURL = "http://localhost:7000";
    private const float retryDelay = 3.0f;

    private const float tracking_confidence = 1.0f;
    private const float detection_confidence = 1.0f;
    private const string pShutdownEndpoint = "/shutdown";
    private const string healthCheckEndpoint = "/health";
    private const string cvProcessEndpoint = "/process";
    private const string ccCaptureEndpoint = "/capture_and_send";
    private const string cvSettingsEndpoint = "/settings";
    private const string computerVisionServerURL = "http://localhost:8000";
    private const string captureCameraServerURL = "http://localhost:8001";
    private const string cvName = "Computer Vision Server";
    private const string ccName = "Capture Camera Server";

    private async void Start()
    {
        httpListenerSetup();
        isRunning = true;
        listenerThread = new Thread(StartListener);
        listenerThread.Start();
        isCvAndCcRunning = await CheckBothServerStatuses();

        if (isCvAndCcRunning)
        {
            for (int i = 0; i < 10; i++)
            {
                SendImageFrame();
            }
               
        }
    }

    private async Task<bool> CheckBothServerStatuses()
    {
        bool isComputerVisionServerRunning = false;
        bool isCaptureServerRunning = false;

        for (int attempt = 0; attempt < maxAttempts; attempt++)
        {
            var cvStatusTask = CheckServerStatus(computerVisionServerURL, healthCheckEndpoint, cvName);
            var ccStatusTask = CheckServerStatus(captureCameraServerURL, healthCheckEndpoint, ccName);

            try
            {
                var results = await Task.WhenAll(cvStatusTask, ccStatusTask);
                isComputerVisionServerRunning = results[0];
                isCaptureServerRunning = results[1];

                if (isComputerVisionServerRunning && isCaptureServerRunning)
                {
                    Debug.Log("Both servers are running.");
                    return true;
                }
            }
            catch (Exception error)
            {
                Debug.LogError($"Error while checking server statuses: {error.Message}");
            }

            await Task.Delay(TimeSpan.FromSeconds(retryDelay));
            Debug.LogWarning($"Attempt {attempt + 1} to check server statuses failed.");
        }

        if (!isComputerVisionServerRunning)
        {
            Debug.LogError($"{cvName} is not responding after {maxAttempts} attempts.");
        }
        if (!isCaptureServerRunning)
        {
            Debug.LogError($"{ccName} is not responding after {maxAttempts} attempts.");
        }
        return false;
    }

    private async Task<bool> CheckServerStatus(string serverUrl, string healthEndpoint, string serverName)
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
        return false;
    }

    private void httpListenerSetup()
    {
        httpListener = new HttpListener();
        httpListener.Prefixes.Add(unityServerURL + shutdownEndpoint);
        httpListener.Start();
    }

    private void StartListener()
    {
        while (isRunning)
        {
            try
            {
                HttpListenerContext context = httpListener.GetContext();
                ProcessRequest(context);
            }
            catch (HttpListenerException)
            {
                Debug.LogError("HttpListener has encountered an exception");
                break;
            }
            catch (Exception error)
            {
                Debug.LogError("Server error HTTP: " + error.Message);
            }
        }
    }

    private void StopListener()
    {
        isRunning = false;
        if (httpListener != null && httpListener.IsListening)
        {
            httpListener.Stop();
        }
        if (listenerThread != null && listenerThread.IsAlive)
        {
            listenerThread.Join();
            Debug.Log("Server was stopped correctly");
        }
    }
    private void SendShutdownSignal(string serverUrl, string shutdownEndpoint, string serverName)
    {
        var shutdownRequest = (HttpWebRequest)WebRequest.Create(serverUrl + shutdownEndpoint);
        shutdownRequest.Method = "POST";
        try
        {
            using (var response = (HttpWebResponse)shutdownRequest.GetResponse())
            {
                if (response.StatusCode == HttpStatusCode.OK)
                {
                    Debug.Log($"{serverName}: Server notified of shutdown.");
                }
            }
        }
        catch (Exception error)
        {
            Debug.LogError($"{serverName}: Error notifying server about shutdown - {error.Message}");
        }
    }

    private void OnApplicationQuit()
    {
        SendShutdownSignal(computerVisionServerURL, pShutdownEndpoint, cvName);
        SendShutdownSignal(captureCameraServerURL, pShutdownEndpoint, ccName);
        StopListener();
    }

    private void ProcessRequest(HttpListenerContext context)
    {
        HttpListenerRequest request = context.Request;
        HttpListenerResponse response = context.Response;
        if (request.Url.AbsolutePath == pShutdownEndpoint)
        {
            ClientShutdownSignal(response, cvName);
            ClientShutdownSignal(response, ccName);
        }
        response.Close();
    }

    private void ClientShutdownSignal(HttpListenerResponse response, string serverName)
    {
        Debug.Log($"{serverName}: Shutdown signal received.");
        StopListener();
        response.StatusCode = (int)HttpStatusCode.OK;
        response.OutputStream.Close();
        Debug.Log($"{serverName}: Server is shutting down...");
    }

    private async Task<string> StartImageCapture()
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
                        string imageBase64 = await streamReader.ReadToEndAsync();
                        imageBase64 = imageBase64.Substring(1, imageBase64.Length - 2);
                        Debug.Log($"{ccName}: Capture initiated successfully and image received.");
                        return imageBase64;
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

        return null;
    }

    private async void SendImageFrame()
    {
        string imageBase64 = await StartImageCapture();
        if (string.IsNullOrEmpty(imageBase64))
        {
            Debug.LogError($"{cvName}: No image frame to send.");
            return;
        }
        var request = (HttpWebRequest)WebRequest.Create(computerVisionServerURL + cvProcessEndpoint);
        request.Method = "POST";
        request.ContentType = "application/json";
        string jsonPayload = "{\"image_base64\": \"" + imageBase64 + "\"}";

        try
        {
            using (var streamWriter = new StreamWriter(request.GetRequestStream()))
            {
                streamWriter.Write(jsonPayload);
            }
            using (var response = (HttpWebResponse)request.GetResponse())
            {
                using (var streamReader = new StreamReader(response.GetResponseStream()))
                {
                    string result = streamReader.ReadToEnd();
                    Debug.Log($"{cvName}: Processing result received - {result}");
                }
                if (response.StatusCode == HttpStatusCode.OK)
                {
                    Debug.Log($"{cvName}: Got the array with points.");
                }
            }
        }
        catch (Exception error)
        {
            Debug.LogError($"{cvName}: Error from server - {error.Message}");
        }
    }

    private void SendCVSettings(float threshold, string model)
    {
        var request = (HttpWebRequest)WebRequest.Create(computerVisionServerURL + cvSettingsEndpoint);
        request.Method = "POST";
        request.ContentType = "application/json";
        string jsonPayload = "{\"threshold\": " + threshold + ", \"model\": \"" + model + "\"}";
        try
        {
            using (var streamWriter = new StreamWriter(request.GetRequestStream()))
            {
                streamWriter.Write(jsonPayload);
            }
            using (var response = (HttpWebResponse)request.GetResponse())
            {
                using (var streamReader = new StreamReader(response.GetResponseStream()))
                {
                    string result = streamReader.ReadToEnd();
                    Debug.Log($"{cvName}: Settings change response - {result}");
                }
                if (response.StatusCode == HttpStatusCode.OK)
                {
                    Debug.Log($"{cvName}: Settings were changed successfully.");
                }
            }
        }
        catch (Exception error)
        {
            Debug.LogError($"{cvName}: Error changing settings - {error.Message}");
        }
    }
}