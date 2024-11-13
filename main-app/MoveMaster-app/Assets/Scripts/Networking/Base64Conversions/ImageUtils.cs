using System;
using UnityEngine;

public static class ImageUtils
{ 
    public static byte[] Base64ToByteArray(string base64String)
    {
        return System.Convert.FromBase64String(base64String);
    }

    public static Texture2D Base64ToTexture(string base64String)
    {
        if (string.IsNullOrEmpty(base64String))
        {
            Debug.LogError("Base64 string is null or empty.");
            return null;
        }

        byte[] imageBytes;
        try
        {
            imageBytes = Convert.FromBase64String(base64String);
        }
        catch (FormatException ex)
        {
            Debug.LogError($"Base64 string is invalid: {ex.Message}");
            return null;
        }

        if (imageBytes.Length == 0)
        {
            Debug.LogError("Decoded Base64 string results in an empty byte array.");
            return null;
        }

        // Initial size is fine as Unity will resize it during LoadImage
        Texture2D texture = new(1, 1);  
        bool isLoaded = texture.LoadImage(imageBytes);

        if (!isLoaded)
        {
            Debug.LogError("Failed to load image data into texture.");
            return null;
        }

        return texture;
    }

    public static string TextureToBase64(Texture2D texture)
    {
        if (texture == null)
        {
            Debug.LogError("Texture is null.");
            return null;
        }
        
        byte[] imageBytes;
        try
        {
            imageBytes = texture.EncodeToPNG();
            if (imageBytes.Length == 0)
            {
                Debug.LogError("Encoded PNG data is empty.");
                return null;
            }
        }
        catch (Exception ex)
        {
            Debug.LogError($"Failed to encode texture to PNG: {ex.Message}");
            return null;
        }

        return Convert.ToBase64String(imageBytes);
    }
}