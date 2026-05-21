#import <Cocoa/Cocoa.h>
#import <CoreGraphics/CoreGraphics.h>
#import <dlfcn.h>
#import <string.h>

typedef CGImageRef (*CGWindowListCreateImageFn)(
    CGRect screenBounds,
    CGWindowListOption windowListOption,
    CGWindowID windowID,
    CGWindowImageOption imageOption);

int main(int argc, const char* argv[])
{
    const char* out_path = (argc > 1) ? argv[1]
        : "docs/screenshots/desktop-chat.png";

    CFArrayRef list = CGWindowListCopyWindowInfo(
        kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements,
        kCGNullWindowID);
    if (!list) {
        fprintf(stderr, "Failed to list windows\n");
        return 1;
    }

    CGWindowID window_id = 0;
    const CFIndex n = CFArrayGetCount(list);
    for (CFIndex i = 0; i < n; ++i) {
        CFDictionaryRef info = CFArrayGetValueAtIndex(list, i);
        CFStringRef owner = CFDictionaryGetValue(info, kCGWindowOwnerName);
        CFStringRef name = CFDictionaryGetValue(info, kCGWindowName);
        char owner_buf[256] = {0};
        char name_buf[256] = {0};
        if (owner) {
            CFStringGetCString(owner, owner_buf, sizeof(owner_buf), kCFStringEncodingUTF8);
        }
        if (name) {
            CFStringGetCString(name, name_buf, sizeof(name_buf), kCFStringEncodingUTF8);
        }
        if (strcmp(owner_buf, "llm_desktop") == 0 || strstr(name_buf, "LLM Engine") != NULL) {
            CFNumberRef num = CFDictionaryGetValue(info, kCGWindowNumber);
            if (num) {
                CFNumberGetValue(num, kCFNumberIntType, &window_id);
            }
            break;
        }
    }
    CFRelease(list);

    if (window_id == 0) {
        fprintf(stderr, "llm_desktop window not found (is the app running?)\n");
        return 2;
    }

    CGWindowListCreateImageFn create_image = (CGWindowListCreateImageFn)dlsym(
        RTLD_DEFAULT, "CGWindowListCreateImage");
    if (!create_image) {
        fprintf(stderr, "CGWindowListCreateImage not available\n");
        return 3;
    }

    CGImageRef image = create_image(
        CGRectNull,
        kCGWindowListOptionIncludingWindow,
        window_id,
        kCGWindowImageBoundsIgnoreFraming | kCGWindowImageNominalResolution);
    if (!image) {
        fprintf(stderr, "Capture failed (Screen Recording permission may be required)\n");
        return 3;
    }

    NSBitmapImageRep* rep = [[NSBitmapImageRep alloc] initWithCGImage:image];
    CGImageRelease(image);
    NSData* data = [rep representationUsingType:NSBitmapImageFileTypePNG properties:@{}];
    NSString* path = [NSString stringWithUTF8String:out_path];
    if (![data writeToFile:path atomically:YES]) {
        fprintf(stderr, "Failed to write %s\n", out_path);
        return 4;
    }

    printf("Saved %s (%ld bytes, %zu x %zu px)\n",
           out_path,
           (long)data.length,
           (size_t)rep.pixelsWide,
           (size_t)rep.pixelsHigh);
    return 0;
}
