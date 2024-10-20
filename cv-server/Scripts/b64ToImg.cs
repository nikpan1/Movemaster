using UnityEngine;
using System.IO;

public class LoadImage : MonoBehaviour
{
    public string imagePath = "path/to/your/image.png"; // Adjust this path as needed

    void Start()
    {
        StartCoroutine(LoadImageCoroutine());
    }

    System.Collections.IEnumerator LoadImageCoroutine()
    {
        string filePath = "file://" + Application.dataPath + "/" + imagePath;
        WWW www = new WWW(filePath);
        yield return www;

        Texture2D texture = new Texture2D(2, 2);
        www.LoadImageIntoTexture(texture);

        GetComponent<Renderer>().material.mainTexture = texture;
    }
}
