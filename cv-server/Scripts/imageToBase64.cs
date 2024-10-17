using UnityEngine;
using System.IO;

public class SaveImage : MonoBehaviour
{
    public Texture2D texture;

    void Start()
    {
        byte[] bytes = texture.EncodeToPNG();
        File.WriteAllBytes(Application.dataPath + "/SavedImage.png", bytes);
        Debug.Log("Image saved!");
    }
}