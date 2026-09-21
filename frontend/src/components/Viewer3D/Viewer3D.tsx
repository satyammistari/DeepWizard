import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";

export default function Viewer3D() {
  return (
    <Canvas camera={{ position: [80, 80, 80], fov: 50 }} className="h-full w-full">
      <color attach="background" args={["#0b1220"]} />
      <ambientLight intensity={0.45} />
      <directionalLight position={[60, 80, 40]} intensity={1.1} />
      <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
        <planeGeometry args={[100, 100]} />
        <meshStandardMaterial color="#3d6b4f" />
      </mesh>
      <gridHelper args={[100, 20, "#64748b", "#1e293b"]} />
      <OrbitControls makeDefault />
    </Canvas>
  );
}
