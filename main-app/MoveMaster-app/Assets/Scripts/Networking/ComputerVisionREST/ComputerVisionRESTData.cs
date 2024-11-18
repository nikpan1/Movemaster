using UnityEngine;


[System.Serializable]
public class ComputerVisionRESTData
{
    public Vector3[] points = new Vector3[33];
}

[System.Serializable]
public class ComputerVisionRESTRequest
{
    public string base64_image;
}
