using UnityEngine;
using UnityEngine.Events;


public class WebcamImageCaptureHandler : MonoBehaviour
{
    private Texture Image { get; set; }
    private readonly string webcamImageCaptureServerUrl = "http://localhost:8001";
    private readonly WebcamImageCaptureSettings settings = new();

    public UnityEvent<Texture> onNewFrameTriggerTexture;
    public UnityEvent<string> onNewFrameTriggerReceivedJson;

    private RESTBaseServer _baseServer;
    
    private void Awake()
    { 
        _baseServer = new RESTBaseServer();
        _baseServer.StartListener();
        _baseServer.RegisterAction(ApiMethod.POST, "/new_frame", HandleNewFrame);
        _baseServer.RegisterAction(ApiMethod.PUT, "/new_frame", HandleNewFrameReceive);
        
        SendSettings(); 
    }

    private string HandleNewFrameReceive(string s)
    {
        Debug.Log(s);
        return "P";
    }

    private void OnApplicationQuit()
    {
        SendShutdown();
        _baseServer.StopListener();
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

        _ = _baseServer.SendRequest(endpoint, webcamImageCaptureServerUrl, content);
    }

    public void SendSettings()
    {
        RESTEndpoint endpoint = new("/settings", ApiMethod.POST);
        string content = settings.ToJson();

        _ = _baseServer.SendRequest(endpoint, webcamImageCaptureServerUrl, content);
    }
}
