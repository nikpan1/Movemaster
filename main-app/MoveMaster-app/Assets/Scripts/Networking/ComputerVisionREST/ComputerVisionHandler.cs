using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Windows;

public class ComputerVisionHandler : MonoBehaviour
{
    private readonly ComputerVisionSettings settings = new();
    private readonly string webcamImageCaptureServerUrl = "http://localhost:8000";
    [SerializeField] public ComputerVisionData Data; //{ get; private set; }

    private void Start() => SendSettings();
    private void OnApplicationQuit() => SendShutdown();


    public void Update()
    {
        if (UnityEngine.Input.GetKeyDown(KeyCode.Space))
        {
            Debug.Log("Space key was pressed.");
            GetFrame();
        }
    }

    public async void GetFrame()
    {
        RESTEndpoint endpoint = new("/frame", ApiMethod.GET);

        string content = await RESTBaseServer.Instance.SendRequest(endpoint, "http://localhost:8001", null);
        
        Debug.Log(content);
        ComputerVisionRequest request = JsonUtility.FromJson<ComputerVisionRequest>(content);

        endpoint = new("/process", ApiMethod.POST);     
        string result = await RESTBaseServer.Instance.SendRequest(endpoint, webcamImageCaptureServerUrl, content);
        Debug.Log(result);

        Data = ExtractData(result);
    }

    public async void SendFrame(string base64frame)
    {
        // this should we optimize
        RESTEndpoint endpoint = new("/process", ApiMethod.POST);
        //string result = await RESTBaseServer.Instance.SendRequest(endpoint, webcamImageCaptureServerUrl, base64frame);
        
        //Data = ExtractData(result);
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

    public ComputerVisionData ExtractData(string jsonString)
    {
        if (string.IsNullOrEmpty(jsonString))
        {
            Debug.LogError("ExtractData received a null or empty JSON string.");
            return null;
        }

        // Deserialize the JSON string into ComputerVisionData
        ComputerVisionData extractedData = JsonUtility.FromJson<ComputerVisionData>(jsonString);

        // Now you can access the points array
        for (int i = 0; i < 33; i++)
        {
            Vector3 point = extractedData.points[i];
            Debug.Log($"Point {i + 1}: {point.x}, {point.y}, {point.z}");
        }
        return extractedData;
    }
}
