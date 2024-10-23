using System;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using CVServer.Models;
using System.Text;
using System.Collections.Generic;

public class CVServerJsonData
{
    public List<List<float>> positions { get; set; }
}

public class ComputerVisionServer : MonoBehaviour
{
    private HttpListener httpListener;
    private Thread listenerThread;
    private bool isRunning = false;
    private JointPosition[] joints = new JointPosition[33];
    private const float retryDelay = 3.0f;
    private const int maxAttempts = 10;
    private static readonly HttpClient client = new HttpClient();
    private const string shutdownEndpoint = "/shutdown/";
    private const string unityServerURL = "http://localhost:7000";

    private bool isCVSRunning = false;
    private const float tracking_confidence = 1.0f;
    private const float detection_confidence = 1.0f;
    private const string cvProcessEndpoint = "/process";
    private const string cvSettingsEndpoint = "/settings";
    private const string cvShutdownEndpoint = "/shutdown";
    private const string cvHealthCheckEndpoint = "/health";
    private const string computerVisionServerURL = "http://localhost:8000";

    private async void Start()
    {
        httpListenerSetup();
        isRunning = true;
        _ = StartListener();
        isCVSRunning = await CheckServerStatus();
        SendCVSettings(detection_confidence, tracking_confidence);
    }


    private async Task<bool> CheckServerStatus()
    {
        for (int attempt = 0; attempt < maxAttempts; attempt++)
        {
            try
            {
                HttpResponseMessage statusResponse = await client.GetAsync(computerVisionServerURL + cvHealthCheckEndpoint);
                if (statusResponse.IsSuccessStatusCode)
                {
                    return true;
                }
            }
            catch (Exception error)
            {
                Debug.LogWarning("Attempt to coonect to CVServer failed: " + error.Message);
            }
            await Task.Delay(TimeSpan.FromSeconds(retryDelay));
        }
        return false;
    }

    private void httpListenerSetup()
    {
        httpListener = new HttpListener();
        httpListener.Prefixes.Add(unityServerURL + shutdownEndpoint);
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

    private async Task SendShutdownSignal()
    {
        using (var content = new StringContent("Unity server is shutting down", Encoding.UTF8, "application/json" ))
        {
            try
            {
                HttpResponseMessage shutdownResponse = await client.PostAsync(computerVisionServerURL + cvShutdownEndpoint, content);
                shutdownResponse.EnsureSuccessStatusCode();
            }
            catch (Exception ex)
            {
                Debug.Log(ex);
            }

        }
    }

    private async void OnApplicationQuit()
    {
        if (isCVSRunning)
        {
            await SendShutdownSignal();
        }
        StopListener();
    }



    private void ProcessRequest(HttpListenerContext context)
    {
        HttpListenerRequest request = context.Request;
        HttpListenerResponse response = context.Response;
        if (request.Url.AbsolutePath == shutdownEndpoint)
        {
            ClientShutdownSignal(response);
        }
        response.Close();
    }

    private void ClientShutdownSignal(HttpListenerResponse response)
    {
        Debug.Log("Shutdown signal from computer vision server");
        isCVSRunning = false;
        StopListener();
        response.StatusCode = (int)HttpStatusCode.OK;
        response.OutputStream.Close();
        Debug.Log("Server is shutting down...");
    }

    private void SendImageFrame(string imageBase64)
    {
        HttpWebRequest request = (HttpWebRequest)WebRequest.Create(computerVisionServerURL + cvProcessEndpoint);
        request.Method = "POST";
        request.ContentType = "application/json";
        string jsonPayload = "{\"image_base64\": \"}" + imageBase64 + "\"}";
        try
        {
            using (StreamWriter streamWriter = new StreamWriter(request.GetRequestStream()))
            {
                streamWriter.Write(jsonPayload);
            }
            using (HttpWebResponse response = (HttpWebResponse)request.GetResponse())
            {
                using (StreamReader streamReader = new StreamReader(response.GetResponseStream()))
                {
                    string result = streamReader.ReadToEnd();
                    CVServerJsonData data = JsonUtility.FromJson<CVServerJsonData>(result);
                    Debug.Log(data);
                }
                if (response.StatusCode == HttpStatusCode.OK)
                {
                    Debug.Log("Got the array with points");
                }
            }
        }
        catch (Exception error)
        {
            Debug.LogError("Error from computer vision server: " + error.Message);
        }
    }

    private void SendCVSettings(float detection_confidence, float tracking_confidence)
    {
        HttpWebRequest request = (HttpWebRequest)WebRequest.Create(computerVisionServerURL + cvSettingsEndpoint);
        request.Method = "POST";
        request.ContentType = "application/json";
        string jsonPayload = "{\"detection_confidence\": " + detection_confidence + ", \"tracking_confidence\": \"" + tracking_confidence + "\"}";
        try
        {
            using (StreamWriter streamWriter = new StreamWriter(request.GetRequestStream()))
            {
                streamWriter.Write(jsonPayload);
            }
            using (HttpWebResponse response = (HttpWebResponse)request.GetResponse())
            {
                using (StreamReader streamReader = new StreamReader(response.GetResponseStream()))
                {
                    string result = streamReader.ReadToEnd();
                    Debug.Log(result);
                }
                if (response.StatusCode == HttpStatusCode.OK)
                {
                    Debug.Log("Settings were changed");
                }
            }
        }
        catch (Exception error)
        {
            Debug.LogError("Error from computer vision server: " + error.Message);
        }
    }




}
