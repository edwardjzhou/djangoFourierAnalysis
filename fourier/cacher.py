import hashlib
import functools

from django.core.cache import cache
from django import http
from django.utils.encoding import force_bytes, iri_to_uri



def json_response_cache_page_decorator(seconds):
    """Cache only when there's a healthy http.JsonResponse response."""

    def decorator(func):

        @functools.wraps(func)
        def inner(request, *args, **kwargs):
            cache_key = 'json_response_cache:{}:{}'.format(
                func.__name__,
                hashlib.md5(force_bytes(iri_to_uri(
                    request.build_absolute_uri()
                ))).hexdigest()
            )
            content = cache.get(cache_key)
            if content is not None:
                return http.HttpResponse(
                    content,
                    content_type='application/json'
                )
            response = func(request, *args, **kwargs)
            if (
                isinstance(response, http.JsonResponse) and
                response.status_code in (200, 304)
            ):
                cache.set(cache_key, response.content, seconds)
            return response

        return inner

    return decorator