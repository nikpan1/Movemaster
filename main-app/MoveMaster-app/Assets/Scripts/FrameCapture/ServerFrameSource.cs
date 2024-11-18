using System.Collections;
using System.Net.Http;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Events;


public class ServerFrameSource : IFrameSource
{
    private Texture2D LatestFrame { get; set; }
    private const string WebcamImageCaptureServerUrl = "http://localhost:8001";
    private readonly WebcamImageCaptureRESTSettings _settings = new();
    private readonly int _framerate = 25;
    
    private UnityEvent<Texture2D> _onNewFrameTriggerTexture;

    private RestBaseServer _baseServer;
    private bool _isCapturing = false;
    
    public void SetupCapture()
    { 
        _baseServer = new();
        _baseServer.StartListener(); 
        
        SendSettings();
        _isCapturing = true;
    }
    
    public void CleanupCapture()
    {
        SendShutdown();
        
        _isCapturing = false;
        _baseServer.StopListener();
    }

    public IEnumerator RunCapture(UnityEvent<Texture2D> trigger)
    {
        _onNewFrameTriggerTexture = trigger;
        RESTEndpoint getFrame = new("/new_frame", HttpMethod.Get);
        while (_isCapturing)
        {
            _ = RetrieveNewFrame(getFrame);
            
            yield return new WaitForSeconds(1 / _framerate);
        }
    }

    private async Task RetrieveNewFrame(RESTEndpoint getFrame)
    {
        string content = await _baseServer.SendRequest(getFrame, WebcamImageCaptureServerUrl, "");
        HandleNewFrame(content);
    }
    
    private void HandleNewFrame(string input)
    {
        if (string.IsNullOrEmpty(input))
        {
            return;
        } 
        
        // Needed in order to fix json str structure 
        input = input.Replace("\\\"", "\"");
        
        ServerFrameInputStructure request = JsonUtility.FromJson<ServerFrameInputStructure>(input.Replace("\\\"", "\""));
        string base64String = request.base64_image.Trim('"');
        LatestFrame = ImageUtils.Base64ToTexture(base64String);
        
        _onNewFrameTriggerTexture?.Invoke(LatestFrame);
    }

    private void SendShutdown()
    {
        RESTEndpoint endpoint = new("/shutdown", HttpMethod.Delete);
        string content = "";

        _ = _baseServer.SendRequest(endpoint, WebcamImageCaptureServerUrl, content);
    }

    private void SendSettings()
    {
        RESTEndpoint endpoint = new("/settings", HttpMethod.Post);
        string content = _settings.ToJson();

        _ = _baseServer.SendRequest(endpoint, WebcamImageCaptureServerUrl, content);
    }
}
