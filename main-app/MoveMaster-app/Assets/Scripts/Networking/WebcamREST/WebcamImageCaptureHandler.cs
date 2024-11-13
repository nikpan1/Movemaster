using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;


public class WebcamImageCaptureHandler : MonoBehaviour
{
    public Texture Image { get; private set; }
    private readonly string webcamImageCaptureServerUrl = "http://localhost:8001";
    private readonly WebcamImageCaptureSettings settings = new();

    public UnityEvent<Texture> onNewFrameTriggerTexture;
    public UnityEvent<string> onNewFrameTriggerReceivedJson;

    private void Start() => SendSettings();
    private void OnApplicationQuit() => SendShutdown();

    private void Awake()
    {
        RESTBaseServer.Instance.RegisterAction(ApiMethod.POST, "/new_frame", HandleNewFrame);
    }

    private string HandleNewFrame(string input)
    {
        ComputerVisionRequest request = JsonUtility.FromJson<ComputerVisionRequest>(input);
        string base64String = request.image_base64.Trim('"');
 

        Image = ImageUtils.Base64ToTexture(base64String);
        onNewFrameTriggerTexture?.Invoke(Image);

        onNewFrameTriggerReceivedJson?.Invoke(input);

        return "{\"status\": \"OK\"}";
    }

    private void SendShutdown()
    {
        RESTEndpoint endpoint = new("/shutdown", ApiMethod.DELETE);
        string content = "";

        _ = RESTBaseServer.Instance.SendRequest(endpoint, webcamImageCaptureServerUrl, content);
    }

    public void SendSettings()
    {
        RESTEndpoint endpoint = new("/settings", ApiMethod.POST);
        string content = settings.ToJson();

        _ = RESTBaseServer.Instance.SendRequest(endpoint, webcamImageCaptureServerUrl, content);
    }
}
