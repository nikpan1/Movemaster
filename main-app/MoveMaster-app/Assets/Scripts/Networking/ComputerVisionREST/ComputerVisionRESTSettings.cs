using UnityEngine;

[System.Serializable]
public class ComputerVisionRESTSettings
{
    public float min_detection_confidence = 0.7f;
    public float min_tracking_confidence = 0.7f;

    public string ToJson()
    {
        return JsonUtility.ToJson(this);
    }
}

