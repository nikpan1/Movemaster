using System.Collections;
using System.Net.Http;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Events;

public class ServerFrameSource : IFrameSource
{
    private const string WebcamImageCaptureServerUrl = "http://localhost:8001";
    private readonly WebcamImageCaptureRESTSettings _settings = new();
    private RestBaseServer _baseServer;
    private const int Framerate = 20;

    private bool _isCapturing;
    private Texture2D _latestFrame;

    private UnityEvent<Texture2D> _onNewFrameTriggerTexture;

    public void SetupCapture()
    {
        _baseServer = new RestBaseServer();
        _baseServer.StartListener();
        
        // Note: This should be not commented in production code
        //SendSettings();
        
        _isCapturing = true;
    }

    public void CleanupCapture()
    {
        // Note: This should be not commented in production code
        //SendShutdown();

        _isCapturing = false;
        _baseServer.StopListener();
    }

    public IEnumerator RunCapture(UnityEvent<Texture2D> triggers)
    {
        _onNewFrameTriggerTexture = triggers;
        RESTEndpoint getFrame = new("/new_frame", HttpMethod.Get);
        while (_isCapturing)
        {
            _ = RetrieveNewFrame(getFrame);

            yield return new WaitForSeconds(1 / Framerate);
        }
    }

    private async Task RetrieveNewFrame(RESTEndpoint getFrame)
    {
        var content = await _baseServer.SendRequest(getFrame, WebcamImageCaptureServerUrl, "");
        HandleNewFrame(content);
    }

    private void HandleNewFrame(string input)
    {
        if (string.IsNullOrEmpty(input)) return;

        // Needed in order to fix json str structure 
        input = input.Replace("\\\"", "\"");

        var request = JsonUtility.FromJson<ServerFrameInputStructure>(input.Replace("\\\"", "\""));
        var base64String = request.base64_image.Trim('"');
        _latestFrame = ImageUtils.Base64ToTexture2D(base64String);

        _onNewFrameTriggerTexture?.Invoke(_latestFrame);
    }

    private void SendShutdown()
    {
        RESTEndpoint endpoint = new("/shutdown", HttpMethod.Delete);
        var content = "";

        _ = _baseServer.SendRequest(endpoint, WebcamImageCaptureServerUrl, content);
    }

    private void SendSettings()
    {
        RESTEndpoint endpoint = new("/settings", HttpMethod.Post);
        var content = _settings.ToJson();

        _ = _baseServer.SendRequest(endpoint, WebcamImageCaptureServerUrl, content);
    }
}