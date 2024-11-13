using System;
using System.Net.Http;

[Serializable]
public enum ApiMethod
{
    GET,
    POST,
    PUT,
    DELETE
}

public static class ApiMethodExtensions
{
    public static ApiMethod ToApiMethod(this string method)
    {
        return method.ToUpper() switch
        {
            "GET" => ApiMethod.GET,
            "POST" => ApiMethod.POST,
            "PUT" => ApiMethod.PUT,
            "DELETE" => ApiMethod.DELETE,
            _ => throw new ArgumentException("Invalid method string"),
        };
    }

    public static HttpMethod ToHttpMethod(this ApiMethod method)
    {
        return method switch
        {
            ApiMethod.GET => HttpMethod.Get,
            ApiMethod.POST => HttpMethod.Post,
            ApiMethod.PUT => HttpMethod.Put,
            ApiMethod.DELETE => HttpMethod.Delete,
            _ => throw new ArgumentException("Invalid ApiMethod enum value"),
        };
    }
}