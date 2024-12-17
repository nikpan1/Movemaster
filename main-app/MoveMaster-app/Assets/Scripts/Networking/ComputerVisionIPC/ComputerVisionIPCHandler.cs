using System;
using System.Threading;
using System.Threading.Tasks;
// using AsyncIO;
// using NetMQ;
// using NetMQ.Sockets;
// using Newtonsoft.Json;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.Serialization;


public class ComputerVisionIPCHandler : MonoBehaviour//, IDisposable
{
    #region ZeroMQ IPC
    // private RequestSocket _client;

    private readonly string _protocol = "tcp";
    private readonly string _ip = "127.0.0.1";
    private readonly int _port = 5556;

    private string _serverAddress;
    #endregion

    #region Flow Control
    private readonly SemaphoreSlim _semaphore = new(5);
    private CancellationTokenSource _cancellationTokenSource = new();
    private bool _isActive;

    // @TODO: should be adjusted as it goes with TASK-30
    private DateTime _lastRequestTime = DateTime.MinValue;
    private readonly TimeSpan _minInterval = TimeSpan.FromMilliseconds(333);
    #endregion
    
    [SerializeField] private Vector3[] latestResult;
    //
    // private void Start() => SetupIpcHandler();
    // private void OnApplicationQuit() => CleanupIpcHandler();
    //     
    // private void SetupIpcHandler()
    // {
    //     _serverAddress = $"{_protocol}://{_ip}:{_port}";
    //     
    //     _isActive = true;
    //     // ForceDotNet.Force();
    //     // NetMQConfig.Linger = new TimeSpan(0, 0, 1);
    //     // _client = new RequestSocket();
    //     // _client.Options.Linger = new TimeSpan(0, 0, 1);
    //     // _client.Connect(_serverAddress);
    // }
    //
    // public Vector3[] GetLatestResult()
    // {
    //     return latestResult;
    // }
    //
    // public void ProcessImage(Texture2D image)
    // {
    //     if (DateTime.Now - _lastRequestTime < _minInterval)
    //         return; // Skip if it's too soon to process again
    //
    //     _lastRequestTime = DateTime.Now;
    //
    //     // Convert the image to bytes, it needs to be done on the main Thread
    //     var textureBytes = image.EncodeToPNG();
    //     
    //     // Start processing the image asynchronously
    //     _ = Task.Run(() => ProcessImageAsync(textureBytes));
    // }
    //
    // private async Task ProcessImageAsync(byte[] textureBytes)
    // {
    //     // Cancel the previous request
    //     _cancellationTokenSource.Cancel();
    //     _cancellationTokenSource = new CancellationTokenSource();
    //     var token = _cancellationTokenSource.Token;
    //
    //     // Ensure only one task executes at a time
    //     await _semaphore.WaitAsync();
    //     try
    //     {
    //         if (token.IsCancellationRequested)
    //         { 
    //             return; // Checking if this request was canceled before starting
    //         }
    //
    //         var base64Image = ImageUtils.Texture2DToBase64(textureBytes);
    //         var json = $"{{\"base64_image\":\"{base64Image}\"}}";
    //
    //         _client.SendFrame(json);
    //         if (_client.TryReceiveFrameString(TimeSpan.FromSeconds(0.5f), out var result))
    //         {
    //             if (token.IsCancellationRequested)
    //             {
    //                 //Debug.Log("ProcessImageAsync: Response ignored due to cancellation.");
    //                 return;
    //             }
    //
    //             ComputerVisionIPCData pointData = JsonConvert.DeserializeObject<ComputerVisionIPCData>(result);
    //             var temppointsArray = new Vector3[pointData.points.Length];
    //             for (int i = 0; i < pointData.points.Length; i++)
    //             {
    //                 temppointsArray[i] = new Vector3(
    //                     (float)pointData.points[i][0],
    //                     (float)pointData.points[i][1],
    //                     (float)pointData.points[i][2] 
    //                 );
    //             }
    //             latestResult = temppointsArray;
    //         }
    //     }
    //     catch (OperationCanceledException)
    //     {
    //         //Debug.Log("ProcessImageAsync: Operation was canceled.");
    //     }
    //     catch (Exception ex)
    //     {
    //         //Debug.LogError($"ProcessImageAsync encountered an error: {ex}");
    //     }
    //     finally
    //     {
    //         _semaphore.Release();
    //     }
    // }
    //
    // private void CleanupIpcHandler()
    // {
    //     _cancellationTokenSource?.Cancel();
    //     _isActive = false;
    //
    //     // Ensure _client is not already disposed before disposing
    //     if (_client != null)
    //     {
    //         try
    //         {
    //             _client.Dispose();
    //         }
    //         catch (ObjectDisposedException)
    //         { 
    //             // Do nothing
    //         }
    //         _client = null;
    //     }
    //
    //     NetMQConfig.Cleanup(false);
    // }
    //
    //
    // #region Dispose pattern implementation
    //
    // private void OnDisable()
    // {
    //     CleanupIpcHandler();
    // }
    //
    // public void Dispose()
    // {
    //     CleanupIpcHandler();
    // }
    //
    // #endregion
}