using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

[System.Serializable]
public class CameraInfo
{
    public int id;
    public string name;
}

[System.Serializable]
public class CameraListResponse
{
    public List<CameraInfo> cameras;
}

public class CameraSelectionDropdown : MonoBehaviour
{
    [SerializeField] private TMP_Dropdown dropdown;
    private const string pythonServerUrl = "http://localhost:8001";

    private void Awake()
    {
        StartCoroutine(GetCameraList());
    }

    private IEnumerator GetCameraList()
    {
        using (UnityEngine.Networking.UnityWebRequest request =
               UnityEngine.Networking.UnityWebRequest.Get(pythonServerUrl + "/list_cameras"))
        {
            yield return request.SendWebRequest();

            if (request.result == UnityEngine.Networking.UnityWebRequest.Result.Success)
            {
                string json = request.downloadHandler.text;
                CameraListResponse cameraList = JsonUtility.FromJson<CameraListResponse>(json);

                PopulateDropdown(cameraList.cameras);
            }
            else
            {
                Debug.LogError("Error fetching camera list: " + request.error);
            }
        }
    }

    private void PopulateDropdown(List<CameraInfo> cameras)
    {
        dropdown.ClearOptions();

        List<string> options = new List<string>();
        foreach (var cam in cameras)
        {
            options.Add($"{cam.name} (id: {cam.id})");
        }

        dropdown.AddOptions(options);
        dropdown.onValueChanged.AddListener(index => OnCameraSelected(index, cameras));
    }

    private void OnCameraSelected(int selectedIndex, List<CameraInfo> cameras)
    {
        CameraInfo chosenCamera = cameras[selectedIndex];
        Debug.Log($"Selected camera: {chosenCamera.name} with id {chosenCamera.id}");

        StartCoroutine(SetCameraOnServer(chosenCamera.id));
    }

    private IEnumerator SetCameraOnServer(int deviceIndex)
    {
        string jsonBody = "{\"device_index\":" + deviceIndex + "}";
        byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(jsonBody);

        using (var request = new UnityEngine.Networking.UnityWebRequest(pythonServerUrl + "/set_camera", "POST"))
        {
            request.uploadHandler = new UnityEngine.Networking.UploadHandlerRaw(bodyRaw);
            request.downloadHandler = new UnityEngine.Networking.DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");

            yield return request.SendWebRequest();

            if (request.result == UnityEngine.Networking.UnityWebRequest.Result.Success)
            {
                Debug.Log("Successfully changed camera on Python server");
            }
            else
            {
                Debug.LogError("Error setting camera on Python server: " + request.error);
            }
        }
    }
}
