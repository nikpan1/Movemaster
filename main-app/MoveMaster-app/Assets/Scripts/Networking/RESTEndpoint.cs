using System;
using System.Net.Http;

[Serializable]
public class RESTEndpoint
{
    public RESTEndpoint(string url, string method)
    {
        Url = url;
        Method = new HttpMethod(method);
    }

    public RESTEndpoint(string url, HttpMethod method)
    {
        Url = url;
        Method = method;
    }

    public string Url { get; }
    public HttpMethod Method { get; }

    #region Dictionary Key Logic for RESTEndpoint

    public override bool Equals(object obj)
    {
        if (obj is RESTEndpoint other) return Url == other.Url && Method == other.Method;
        return false;
    }

    public override int GetHashCode()
    {
        return HashCode.Combine(Url, Method);
    }

    #endregion
}