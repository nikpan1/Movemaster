using System;
using UnityEngine;

[Serializable]
public class WebcamImageCaptureRESTSettings
{
    // @TODO: Probably will change accordingly to TASK-30
    // Note: SendSettings() needs to be called, preferable from void Start()
    // Note: variables need to be public

    public string ToJson()
    {
        return JsonUtility.ToJson(this);
    }
}