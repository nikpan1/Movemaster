using System;
using UnityEngine;

public static class ImageUtils
{ 
    public static string Texture2DToBase64(byte[] textureBytes)
    { 
        // Convert the byte array to a Base64 string
        string base64String = Convert.ToBase64String(textureBytes);

        return base64String; 
    }
    
    public static Texture2D Base64ToTexture2D(string base64String, Texture2D reusableTexture = null)
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

        if (reusableTexture == null)
        {
            // Create a new Texture2D if none is provided
            reusableTexture = new Texture2D(1, 1);
        }

        bool isLoaded = reusableTexture.LoadImage(imageBytes);

        if (!isLoaded)
        {
            Debug.LogError("Failed to load image data into texture.");
            return null;
        }

        return reusableTexture;
    }
}