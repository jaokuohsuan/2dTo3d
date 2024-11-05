// 初始化 Three.js 預覽
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, 600 / 400, 0.1, 1000);
const renderer = new THREE.WebGLRenderer();
renderer.setSize(600, 400);
document.getElementById('preview').appendChild(renderer.domElement);

camera.position.z = 5;

function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
}
animate();

// 處理上傳按鈕點擊事件
document.getElementById('uploadButton').addEventListener('click', function() {
    const fileInput = document.getElementById('upload');
    const file = fileInput.files[0];
    if (file) {
        const formData = new FormData();
        formData.append('file', file);

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            console.log('data===', data);
            // 使用伺服器返回的文件路徑
            loadModel(data.file_path);
        })
        .catch(error => {
            console.error('Error:', error);
        });
    } else {
        alert('請選��一張照片上傳');
    }
});

// 加載和顯示3D模型
function loadModel(modelPath) {
    const loader = new THREE.PLYLoader();
    loader.load(modelPath, function (geometry) {
        geometry.computeVertexNormals();
        const material = new THREE.MeshStandardMaterial({ color: 0x0055ff });
        const mesh = new THREE.Mesh(geometry, material);
        scene.add(mesh);
    }, undefined, function (error) {
        console.error('An error happened while loading the model:', error);
    });
} 