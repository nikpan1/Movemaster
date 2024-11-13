using UnityEngine;


[System.Serializable]
public class ComputerVisionData
{
    public Vector3[] points = new Vector3[33];
}

[System.Serializable]
public class ComputerVisionRequest
{
    public string image_base64;
}
