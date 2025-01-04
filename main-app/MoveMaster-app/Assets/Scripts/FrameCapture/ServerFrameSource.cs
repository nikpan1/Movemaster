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
    private UnityEvent<ExerciseInference> _triggersExerciseInference; 
    
    
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

    public IEnumerator RunCapture(UnityEvent<Texture2D> triggersTexture2D, UnityEvent<ExerciseInference> triggersExerciseInference)
    {
        _onNewFrameTriggerTexture = triggersTexture2D;
        _triggersExerciseInference = triggersExerciseInference;

        RESTEndpoint getFrame = new("/new_frame", HttpMethod.Get);

        while (_isCapturing)
        {
            var startTime = Time.time;

            // Fetch and process a new frame
            yield return RetrieveNewFrameCoroutine(getFrame);

            // Calculate elapsed time and adjust delay to maintain framerate
            float elapsedTime = Time.time - startTime;
            float delay = Mathf.Max(0, (1 / Framerate) - elapsedTime);
            yield return new WaitForSeconds(delay);
        }
    }

    private IEnumerator RetrieveNewFrameCoroutine(RESTEndpoint getFrame)
    {
        Task<string> fetchTask = RetrieveNewFrame(getFrame);

        while (!fetchTask.IsCompleted)
            yield return null; // Wait for the task to complete

        if (fetchTask.Result != null)
            HandleNewFrame(fetchTask.Result);
    }


    private async Task<string> RetrieveNewFrame(RESTEndpoint getFrame)
    {
        var content = await _baseServer.SendRequest(getFrame, WebcamImageCaptureServerUrl, "");
        return content; // Ensure the method explicitly returns a string
    }

    private void HandleNewFrame(string input)
    {
        if (string.IsNullOrEmpty(input)) return;

        // Needed in order to fix json str structure 
        input = input.Replace("\\\"", "\"");

        var request = JsonUtility.FromJson<ServerFrameInputStructure>(input.Replace("\\\"", "\""));
        var base64String = request.base64_image.Trim('"');
        _latestFrame = ImageUtils.Base64ToTexture2D(base64String, _latestFrame);
        
        var latestPredictedClass = request.latest_predicted_class.Trim('"');
        var latestPredictedConfidence = request.latest_predicted_confidence;
        ExerciseInference ex = new ExerciseInference(latestPredictedClass, latestPredictedConfidence);

        _onNewFrameTriggerTexture?.Invoke(_latestFrame);
        _triggersExerciseInference?.Invoke(ex);
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