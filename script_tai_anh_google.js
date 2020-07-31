//Tải thư viện jquery
var script = document.createElement('script');
script.src = "https://ajax.googleapis.com/ajax/libs/jquery/2.2.0/jquery.min.js";
document.getElementsByTagName('head')[0].appendChild(script);
//Lấy các đường dẫn url của ảnh
var urls = $('.rg_di .rg_meta').map(function() { return JSON.parse($(this).text()).ou; });
//Xuất đường dẫn ra tệp mới, mỗi đường dẫn 1 dòng
var textToSave = urls.toArray().join('\n');
var hiddenElement = document.createElement('a');
hiddenElement.href = 'data:attachment/text,' + encodeURI(textToSave);
hiddenElement.target = '_blank';
hiddenElement.download = 'duongdananh.txt';
hiddenElement.click();
