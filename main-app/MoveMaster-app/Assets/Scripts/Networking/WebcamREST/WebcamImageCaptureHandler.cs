using System.Collections;
using System.Net.Http;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Events;


public class WebcamImageCaptureHandler : MonoBehaviour
{
    private Texture2D LatestFrame { get; set; }
    private const string WebcamImageCaptureServerUrl = "http://localhost:8001";
    private readonly WebcamImageCaptureSettings _settings = new();
    [SerializeField] private int framerate = 25; //  @TODO - this should be moved along with TASK-30 
    
    public UnityEvent<Texture2D> onNewFrameTriggerTexture;
    public UnityEvent<string> onNewFrameTriggerReceivedJson;

    private RestBaseServer _baseServer;
    private bool _isCapturing = false;
    public string content;
    
    private void StopCapturing() => _isCapturing = false;
    private void StartCapturing() => _isCapturing = true;
    
    private void Awake()
    { 
        _baseServer = new();
        _baseServer.StartListener(); 
        
        SendSettings();
        StartCapturing();
        StartCoroutine(ContinousImageCapture());
    }
    
    private void OnApplicationQuit()
    {
        SendShutdown();
        StopCapturing();
        _baseServer.StopListener();
    }
    
    private IEnumerator ContinousImageCapture()
    {
        RESTEndpoint getFrame = new("/new_frame", HttpMethod.Get);
        while (_isCapturing)
        {
            _ = RetrieveNewFrame(getFrame);
            
            yield return new WaitForSeconds(1 / framerate);
        }
    }

    public async Task RetrieveNewFrame(RESTEndpoint getFrame)
    {
        content = await _baseServer.SendRequest(getFrame, WebcamImageCaptureServerUrl, "");
        Debug.Log("Retrieved New Frame");
        HandleNewFrame(content);
    }
    
    private void HandleNewFrame(string input)
    {
        if (string.IsNullOrEmpty(input))
            return; 
        input = input.Replace("\\\"", "\"");
        
        ComputerVisionRequest request = JsonUtility.FromJson<ComputerVisionRequest>(input);
        string base64String = request.base64_image.Trim('"');
        LatestFrame = ImageUtils.Base64ToTexture(base64String);
        
        onNewFrameTriggerReceivedJson?.Invoke(input);
        onNewFrameTriggerTexture?.Invoke(LatestFrame);
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
