window.onload = function() {
    var anchors = document.querySelectorAll('a.external');
    for (var i = 0; i < anchors.length; i++) {
        anchors[i].setAttribute('target', '_blank');
    }
};