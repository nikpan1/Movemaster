using System;

[Serializable]
public class ServerFrameInputStructure
{
    public string base64_image;
    public string latest_predicted_class;
    public float latest_predicted_confidence;
}