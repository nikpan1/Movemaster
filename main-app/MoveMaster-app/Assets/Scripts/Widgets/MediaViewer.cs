using UnityEngine;
using UnityEngine.UI;

public class MediaViewer : MonoBehaviour
{
    private RawImage _rawImage;

    private Sprite _sprite;
    private Rect _rect;
    private Vector2 _pivot = new Vector2(0.5f, 0.5f);
    
    private void Awake()
    {
        _rawImage = GetComponent<RawImage>();
    }

    public void DisplayFrame(Texture texture)
    {
        Texture2D texture2D = texture as Texture2D;
        if (texture2D == null)
        {
            Debug.LogError("The provided texture is not a Texture2D.");
            return;
        }

        _rect = new Rect(0, 0, texture2D.width, texture2D.height);
        _sprite = Sprite.Create(texture2D, _rect, _pivot);
        _rawImage.texture = _sprite.texture;
    }
}