using System;
using UnityEngine;

[Serializable]
public class ComputerVisionRESTData
{
    public Vector3[] points = new Vector3[33];
}

[Serializable]
public class ComputerVisionRESTRequest
{
    public string base64_image;
}