using System.Collections;
using System.Collections.Generic;
using System.Net;
using System.Threading;
using UnityEngine;

public class BaseServer : MonoBehaviour
{
    static BaseServer instance;
    private HttpListener httpListener;
    private Thread listenerThread;

    private void Awake()
    {
        if (instance == null) 
        { 
            instance = new BaseServer(); 
        }
    }
    private void Start()
    {
        listenerThread = new Thread(StartListener);
        listenerThread.Start();
    }

    private void StartListener()
    {

    }
    // Update is called once per frame
    void Update()
    {
        
    }
}
