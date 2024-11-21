/*using System.Net.Http;
using UnityEngine;

public class ComputerVisionRESTHandler : MonoBehaviour
{
    [SerializeField] public ComputerVisionRESTData Data; //{ get; private set; }
    private readonly ComputerVisionRESTSettings settings = new();
    private readonly string webcamImageCaptureServerUrl = "http://localhost:8000";

    private void Start()
    {
        SendSettings();
    }


    public void Update()
    {
        if (Input.GetKeyDown(KeyCode.Space))
        {
            Debug.Log("Space key was pressed.");
            GetFrame();
        }
    }

    private void OnApplicationQuit()
    {
        SendShutdown();
    }

    public async void GetFrame()
    {
        RESTEndpoint endpoint = new("/frame", HttpMethod.Get);

        // string content = await RESTBaseServer.Instance.SendRequest(endpoint, "http://localhost:8001", null);

        // Debug.Log(content);
        // ComputerVisionRequest request = JsonUtility.FromJson<ComputerVisionRequest>(content);

        endpoint = new RESTEndpoint("/process", HttpMethod.Post);
        //string result = await RESTBaseServer.Instance.SendRequest(endpoint, webcamImageCaptureServerUrl, content);
        //Debug.Log(result);

        //Data = ExtractData(result);
    }

    public async void SendFrame(string base64frame)
    {
        RESTEndpoint endpoint = new("/process", HttpMethod.Post);
        string result = await RESTBaseServer.Instance.SendRequest(endpoint, webcamImageCaptureServerUrl, base64frame);

        Data = ExtractData(result);
    }


    private void SendShutdown()
    {
        RESTEndpoint endpoint = new("/shutdown", HttpMethod.Delete);
        var content = "";

        // _ = RESTBaseServer.Instance.SendRequest(endpoint, webcamImageCaptureServerUrl, content);
    }

    public void SendSettings()
    {
        RESTEndpoint endpoint = new("/settings", HttpMethod.Post);
        var content = settings.ToJson();

        // _ = RESTBaseServer.Instance.SendRequest(endpoint, webcamImageCaptureServerUrl, content);
    }

    public ComputerVisionRESTData ExtractData(string jsonString)
    {
        if (string.IsNullOrEmpty(jsonString))
        {
            Debug.LogError("ExtractData received a null or empty JSON string.");
            return null;
        }

        // Deserialize the JSON string into ComputerVisionData
        var extractedData = JsonUtility.FromJson<ComputerVisionRESTData>(jsonString);

        for (var i = 0; i < 33; i++)
        {
            var point = extractedData.points[i];
            Debug.Log($"Point {i + 1}: {point.x}, {point.y}, {point.z}");
        }

        return extractedData;
    }
}*/