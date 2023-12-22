window.onload = function() {
    var anchors = document.querySelectorAll('a.external');
    for (var i = 0; i < anchors.length; i++) {
        anchors[i].setAttribute('target', '_blank');
    }
};

window.onload = function() {
    // Your existing code for external links
    var anchors = document.querySelectorAll('a.external');
    for (var i = 0; i < anchors.length; i++) {
        anchors[i].setAttribute('target', '_blank');
    }

    // New code for images
    var images = document.querySelectorAll('img');
    for (var j = 0; j < images.length; j++) {
        // Wrapping each image in an anchor tag
        var img = images[j];
        var link = document.createElement('a');
        link.setAttribute('href', img.src);
        link.setAttribute('target', '_blank'); // Open in new tab
        img.parentNode.insertBefore(link, img);
        link.appendChild(img);
    }
};
