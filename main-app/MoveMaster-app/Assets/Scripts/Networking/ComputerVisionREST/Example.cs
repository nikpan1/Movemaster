using UnityEngine;
using UnityEngine.UI;

public class Example : MonoBehaviour
{
    public void up(Texture2D texture)
    {
        Debug.Log("hi");

        // Create the sprite from the texture
        Sprite sprite = Sprite.Create(texture, new Rect(0, 0, texture.width, texture.height), Vector2.zero);

        // Set the sprite to the Image component
        Image imageComponent = this.GetComponent<Image>();
        imageComponent.sprite = sprite;

        // Resize the Image's RectTransform to match the texture's aspect ratio
        RectTransform rt = imageComponent.GetComponent<RectTransform>();
        float aspectRatio = (float)texture.width / texture.height;

        // Set width and height while maintaining the aspect ratio
        rt.sizeDelta = new Vector2(rt.sizeDelta.x, rt.sizeDelta.x / aspectRatio);
    }
}
