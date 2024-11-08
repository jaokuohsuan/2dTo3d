// 初始化 Three.js 預覽
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf0f0f0);

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

// 添加 OrbitControls，設置適當的限制
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.screenSpacePanning = true;

// 設置控制範圍
controls.minDistance = 1;
controls.maxDistance = 10;
controls.maxPolarAngle = Math.PI * 0.75; // 限制垂直旋轉角度
controls.minPolarAngle = Math.PI * 0.1;

function animate() {
    requestAnimationFrame(animate);
    controls.update();
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
    // 清除所有現有的模型
    scene.children.forEach((child) => {
        if (child instanceof THREE.Points || child instanceof THREE.Mesh) {
            scene.remove(child);
            if (child.geometry) {
                child.geometry.dispose();
            }
            if (child.material) {
                child.material.dispose();
            }
        }
    });

    const loader = new THREE.PLYLoader();
    loader.load(modelPath, function (geometry) {
        const material = new THREE.PointsMaterial({ 
            size: 0.005,  // 減小點的大小
            vertexColors: true,
            sizeAttenuation: true  // 啟用大小衰減
        });
        const object = new THREE.Points(geometry, material);
        
        // 自動調整模型位置和大小
        geometry.computeBoundingBox();
        const box = geometry.boundingBox;
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        const scale = 2.0 / maxDim;
        object.scale.multiplyScalar(scale);
        object.position.sub(center.multiplyScalar(scale));

        scene.add(object);
        
        // 重置相機位置
        camera.position.set(0, 0, 5);
        controls.reset();
    });
} 