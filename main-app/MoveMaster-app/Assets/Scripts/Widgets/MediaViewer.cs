using System;
using UnityEngine;
using UnityEngine.UI;

public class MediaViewer : MonoBehaviour
{
    private Image _image;
    
    private RectTransform _rectTransform;

    private void Awake()
    {
        _image = GetComponent<Image>();
        _rectTransform = GetComponent<RectTransform>();
    }

    public void DisplayFrame(Texture2D texture)
    { 
        // Create the sprite from the texture
        Sprite sprite = Sprite.Create(texture, new Rect(0, 0, texture.width, texture.height), Vector2.zero);
        _image.sprite = sprite;

        float aspectRatio = (float)texture.width / texture.height;
        _rectTransform.sizeDelta = new Vector2(_rectTransform.sizeDelta.x, _rectTransform.sizeDelta.x / aspectRatio);
    }
}
