using System;

[Serializable]
public class RESTEndpoint
{
    public string Url { get; }
    public ApiMethod Method { get; }

    public RESTEndpoint(string url, ApiMethod method)
    {
        Url = url;
        Method = method;
    }

    #region Equals and GetHashCode for dictionary usage

    public override bool Equals(object obj)
    {
        if (obj is RESTEndpoint other)
        {
            return Url == other.Url && Method == other.Method;
        }
        return false;
    }

    public override int GetHashCode()
    {
        return HashCode.Combine(Url, Method);
    }

    #endregion
}
