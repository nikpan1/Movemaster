using System;
using System.Collections;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Events;
// using OpenCvSharp;


public class WebcamFrameSource : IFrameSource
{
    // private VideoCapture _capture;
    private Texture2D _texture;
    
    private bool _isInitialized = false;
    private bool _isCapturing = false; 
    
    private const int Framerate = 20;
    
    public void SetupCapture()
    {
        Task.Run(() => { 
            Debug.Log("Initializing WebcamFrameSource. Please wait...");
            // @TODO: TASK-30 - `0` selects the default camera. Should be changed
            // _capture = new VideoCapture(0);
            //
            // if (!_capture.IsOpened())
            // {
            //     throw new Exception("WebcamFrameSource is not initialized.");
            // }

            _isInitialized = true;
            Debug.Log("Initializing WebcamFrameSource. Ended :)");
        });
    }

    public void CleanupCapture()
    {
        _isCapturing = false;
        
        // // Release VideoCapture resources.
        // _capture?.Release();
        // _capture?.Dispose();
        // _capture = null;
        //
        _texture = null;
    }

    public IEnumerator RunCapture(UnityEvent<Texture2D> triggers)
    {
        yield return new WaitUntil(() => _isInitialized);
        
        // Create a Texture2D with dimensions matching the captured frames.
        // _texture = new Texture2D(_capture.FrameWidth, _capture.FrameHeight, TextureFormat.RGB24, false); 
        //
        // _isCapturing = true;
        //
        // Mat latestFrame = new Mat();
        // while (_isCapturing && _capture.IsOpened())
        // {
        //     if (!_capture.Read(latestFrame) || latestFrame.Empty())
        //     {
        //         Debug.LogWarning("No frame captured.");
        //         break;
        //     }
        //
        //     MatToTexture(latestFrame, _texture);
        //     triggers?.Invoke(_texture);
        //
        //     yield return new WaitForSeconds(1 / Framerate);
        // }
        //
        // if (_isCapturing)
        // {
        //     throw new Exception("Capturing frames failed.");
        // }
        //
        // latestFrame.Release();
    }

    // private void MatToTexture(Mat mat, Texture2D texture)
    // {
    //     // Convert the Mat to byte array.
    //     byte[] byteArray = mat.ImEncode(".jpg");
    //
    //     // Load the byte array into the texture.
    //     texture.LoadImage(byteArray);
    // }
}
