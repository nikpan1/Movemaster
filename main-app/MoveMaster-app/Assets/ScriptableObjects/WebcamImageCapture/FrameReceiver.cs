using System;
using System.Collections.Concurrent;
using System.IO;
using System.Net;
using System.Threading;
using UnityEngine;

public class FrameReceiver : MonoBehaviour
{
    private HttpListener httpListener;
    private Thread listenerThread;
    private bool isRunning = true;
    private ConcurrentQueue<byte[]> frameQueue = new ConcurrentQueue<byte[]>();
    private Texture2D receivedTexture;

    private void Start()
    {
        listenerThread = new Thread(StartListener);
        listenerThread.Start();
    }

    private void StartListener()
    {
        httpListener = new HttpListener();
        httpListener.Prefixes.Add("http://localhost:5000/send_frame/");
        httpListener.Prefixes.Add("http://localhost:5000/shutdown/");
        httpListener.Start();

        while (isRunning)
        {
            try
            {
                HttpListenerContext context = httpListener.GetContext();
                ProcessRequest(context);
            }
            catch (Exception e)
            {
                Debug.LogError("Server error HTTP: " + e.Message);
            }
        }

        httpListener.Close();
    }

    private void ProcessRequest(HttpListenerContext context)
    {
        HttpListenerRequest request = context.Request;
        HttpListenerResponse response = context.Response;

        if (request.Url.AbsolutePath == "/send_frame")
        {
            byte[] imageData;
            using (var ms = new MemoryStream())
            {
                request.InputStream.CopyTo(ms);
                imageData = ms.ToArray();
            }

            frameQueue.Enqueue(imageData);

            response.StatusCode = (int)HttpStatusCode.OK;
        }
        else if (request.Url.AbsolutePath == "/shutdown")
        {
            Debug.Log("Close sygnal from Python");
            isRunning = false;

            response.StatusCode = (int)HttpStatusCode.OK;
        }

        response.Close();
    }

    private void Update()
    {
        if (frameQueue.TryDequeue(out byte[] imageData))
        {
            if (receivedTexture == null)
            {
                receivedTexture = new Texture2D(2, 2);
            }

            if (receivedTexture.LoadImage(imageData))
            {
                Debug.Log("A new frame was received");
            }
        }
    }

    private void OnGUI()
    {
        if (receivedTexture != null)
        {
            GUI.DrawTexture(new Rect(Convert.ToInt64(1920 / 2), Convert.ToInt64(0), Convert.ToInt64(1920 / 2), Convert.ToInt64(1080)), receivedTexture);
        }
    }

    private void OnApplicationQuit()
    {
        isRunning = false;
        if (httpListener != null && httpListener.IsListening)
        {
            httpListener.Stop();
        }

        if (listenerThread != null && listenerThread.IsAlive)
        {
            listenerThread.Abort();
        }

        SendShutdownToPython();
    }

    private void SendShutdownToPython()
    {
        var shutdownRequest = (HttpWebRequest)WebRequest.Create("http://localhost:5001/unity_shutdown");
        shutdownRequest.Method = "POST";

        try
        {
            using (var response = (HttpWebResponse)shutdownRequest.GetResponse())
            {
                if (response.StatusCode == HttpStatusCode.OK)
                {
                    Debug.Log("Python notified of Unity shutdown.");
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogError("Error notifying Python about shutdown: " + e.Message);
        }
    }
}
