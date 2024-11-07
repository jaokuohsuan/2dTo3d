// 初始化 Three.js 預覽
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf0f0f0);  // 設置背景色為淺灰色

const camera = new THREE.PerspectiveCamera(75, 600 / 400, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(600, 400);
renderer.setPixelRatio(window.devicePixelRatio);
document.getElementById('preview').appendChild(renderer.domElement);

// 添加光源
const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
scene.add(ambientLight);

const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
directionalLight.position.set(1, 1, 1);
scene.add(directionalLight);

// 設置相機位置
camera.position.set(0, 0, 5);

// 添加 OrbitControls
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; // 添加阻尼效果
controls.dampingFactor = 0.05;
controls.screenSpacePanning = false;
controls.minDistance = 1;
controls.maxDistance = 50;
controls.maxPolarAngle = Math.PI / 2;

function animate() {
    requestAnimationFrame(animate);
    controls.update(); // 更新控制器
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
        alert('請選擇一張照片上傳');
    }
});

// 加載和顯示3D模型
function loadModel(modelPath) {
    // 清除之前的模型
    scene.children.forEach((child) => {
        if (child instanceof THREE.Mesh) {
            scene.remove(child);
        }
    });

    const loader = new THREE.PLYLoader();
    loader.load(modelPath, function (geometry) {
        geometry.computeVertexNormals();
        const material = new THREE.MeshStandardMaterial({ 
            color: 0x0055ff,
            roughness: 0.5,
            metalness: 0.1,
        });
        const mesh = new THREE.Mesh(geometry, material);
        
        // 自動調整模型位置和大小
        geometry.computeBoundingBox();
        const box = geometry.boundingBox;
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        const scale = 2.0 / maxDim;
        mesh.scale.multiplyScalar(scale);
        mesh.position.sub(center.multiplyScalar(scale));

        scene.add(mesh);
        
        // 重置相機位置
        camera.position.set(0, 0, 5);
        controls.reset();
    }, (xhr) => {
        console.log((xhr.loaded / xhr.total) * 100 + '% loaded')
    }, function (error) {
        console.error('An error happened while loading the model:', error);
    });
} 