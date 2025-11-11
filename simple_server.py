#!/usr/bin/env python3

import json
import pickle
import uuid
import time
import os
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import threading
import numpy as np
from datetime import datetime

from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from grasp_gen.utils.logging_config import get_logger


class GraspGenHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, grasp_service=None, **kwargs):
        self.grasp_service = grasp_service
        super().__init__(*args, **kwargs)
    
    def do_POST(self):
        try:
            path = urlparse(self.path).path
            
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > 0:
                body = self.rfile.read(content_length)
                try:
                    data = pickle.loads(body)
                except:
                    data = {}
            else:
                data = {}
            
            if path == '/create_session':
                response = self.grasp_service.create_session(data)
            elif path == '/generate_grasps':
                response = self.grasp_service.generate_grasps(data)
            elif path == '/process_pointcloud':
                response = self.grasp_service.process_pointcloud(data)
            elif path == '/health_check':
                response = self.grasp_service.health_check(data)
            elif path == '/delete_session':
                response = self.grasp_service.delete_session(data)
            else:
                response = {"error": f"Unknown endpoint: {path}"}
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/octet-stream')
            self.end_headers()
            self.wfile.write(pickle.dumps(response))
            
        except Exception as e:
            error_response = {"error": str(e)}
            self.send_response(500)
            self.send_header('Content-Type', 'application/octet-stream')
            self.end_headers()
            self.wfile.write(pickle.dumps(error_response))
    
    def log_message(self, format, *args):
        self.grasp_service.logger.info(f"{self.address_string()} - {format % args}")


class SimpleGraspGenService:
    
    def __init__(self, host="0.0.0.0", port=8080):
        self.host = host
        self.port = port
        self.sessions = {}
        self.logger = get_logger("SimpleGraspGenService")
    
    def create_session(self, data):
        try:
            gripper_config = data.get("gripper_config")
            if not gripper_config:
                return {"error": "gripper_config is required"}
            
            if not os.path.exists(gripper_config):
                return {"error": f"Gripper config file not found: {gripper_config}"}
            
            session_id = str(uuid.uuid4())
            
            self.logger.info(f"Loading grasp configuration from {gripper_config}")
            grasp_cfg = load_grasp_cfg(gripper_config)
            
            self.logger.info(f"Creating GraspGenSampler for session {session_id}")
            sampler = GraspGenSampler(grasp_cfg)
            self.sessions[session_id] = sampler
            
            self.logger.info(f"Created session {session_id}")
            return {"session_id": session_id, "status": "success"}
            
        except Exception as e:
            self.logger.error(f"Failed to create session: {str(e)}")
            return {"error": f"Failed to create session: {str(e)}"}
    
    def generate_grasps(self, data):
        try:
            session_id = data["session_id"]
            point_cloud = data["point_cloud"]
            
            if session_id not in self.sessions:
                return {"error": f"Session {session_id} not found"}
            
            grasp_threshold = data.get("grasp_threshold", 0.8)
            num_grasps = data.get("num_grasps", 200)
            topk_num_grasps = data.get("topk_num_grasps", -1)
            min_grasps = data.get("min_grasps", 40)
            max_tries = data.get("max_tries", 6)
            remove_outliers = data.get("remove_outliers", True)
            
            start_time = time.time()
            grasps, grasp_conf = GraspGenSampler.run_inference(
                point_cloud,
                self.sessions[session_id],
                grasp_threshold=grasp_threshold,
                num_grasps=num_grasps,
                topk_num_grasps=topk_num_grasps,
                min_grasps=min_grasps,
                max_tries=max_tries,
                remove_outliers=remove_outliers,
            )
            
            inference_time = time.time() - start_time
            
            import torch
            if isinstance(grasps, torch.Tensor):
                grasps = grasps.cpu().numpy()
            if isinstance(grasp_conf, torch.Tensor):
                grasp_conf = grasp_conf.cpu().numpy()
            
            self.logger.info(f"Generated {len(grasps)} grasps in {inference_time:.3f}s")
            
            return {
                "grasps": grasps,
                "grasp_confidence": grasp_conf,
                "num_grasps": len(grasps),
                "inference_time": inference_time,
                "status": "success"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate grasps: {str(e)}")
            return {"error": f"Failed to generate grasps: {str(e)}"}
    
    def process_pointcloud(self, data):
        try:
            pointcloud_data = data.get("pointcloud_data")
            gripper_config = data.get("gripper_config", "/models/checkpoints/graspgen_franka_panda.yml")
            save_dir = data.get("save_dir", "/models/test_data")
            filename = data.get("filename", None)

            grasp_threshold = data.get("grasp_threshold", 0.7)
            num_grasps = data.get("num_grasps", 100)
            topk_num_grasps = data.get("topk_num_grasps", 20)
            min_grasps = data.get("min_grasps", 40)
            max_tries = data.get("max_tries", 6)
            remove_outliers = data.get("remove_outliers", True)

            if not pointcloud_data:
                return {"error": "pointcloud_data is required"}

            if isinstance(pointcloud_data, dict):
                pc = np.array(pointcloud_data.get("pc", []))
                pc_color = np.array(pointcloud_data.get("pc_color", [])) if "pc_color" in pointcloud_data else None
            else:
                return {"error": "pointcloud_data must be a dict with 'pc' key"}

            if len(pc) == 0:
                return {"error": "Point cloud is empty"}

            self.logger.info(f"Received point cloud with {len(pc)} points")

            session_id = str(uuid.uuid4())
            self.logger.info(f"Loading grasp configuration from {gripper_config}")
            grasp_cfg = load_grasp_cfg(gripper_config)
            self.logger.info(f"Creating GraspGenSampler for session {session_id}")
            sampler = GraspGenSampler(grasp_cfg)

            self.logger.info("Running grasp inference...")
            start_time = time.time()
            grasps, grasp_conf = GraspGenSampler.run_inference(
                pc,
                sampler,
                grasp_threshold=grasp_threshold,
                num_grasps=num_grasps,
                topk_num_grasps=topk_num_grasps,
                min_grasps=min_grasps,
                max_tries=max_tries,
                remove_outliers=remove_outliers,
            )
            inference_time = time.time() - start_time

            import torch
            if isinstance(grasps, torch.Tensor):
                grasps = grasps.cpu().numpy()
            if isinstance(grasp_conf, torch.Tensor):
                grasp_conf = grasp_conf.cpu().numpy()

            self.logger.info(f"Generated {len(grasps)} grasps in {inference_time:.3f}s")

            output_data = {
                "pc": pc.tolist(),
                "grasp_poses": grasps.tolist(),
                "grasp_conf": grasp_conf.tolist(),
            }

            if pc_color is not None and len(pc_color) > 0:
                output_data["pc_color"] = pc_color.tolist()

            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"grasp_result_{timestamp}.json"

            if not filename.endswith('.json'):
                filename += '.json'

            output_path = os.path.join(save_dir, filename)

            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)

            self.logger.info(f"Saved result to {output_path}")

            return {
                "status": "success",
                "num_grasps": len(grasps),
                "grasps": grasps.tolist(),
                "grasp_confidence": grasp_conf.tolist(),
                "inference_time": inference_time,
                "saved_file": output_path,
                "message": f"Successfully processed point cloud and saved to {output_path}"
            }

        except Exception as e:
            self.logger.error(f"Failed to process point cloud: {str(e)}")
            traceback.print_exc()
            return {"error": f"Failed to process point cloud: {str(e)}"}

    def health_check(self, data):
        return {
            "status": "healthy",
            "active_sessions": len(self.sessions),
            "timestamp": time.time()
        }

    def delete_session(self, data):
        try:
            session_id = data["session_id"]
            if session_id not in self.sessions:
                return {"error": f"Session {session_id} not found"}
            
            del self.sessions[session_id]
            self.logger.info(f"Deleted session {session_id}")
            return {"status": "success"}
            
        except Exception as e:
            return {"error": f"Failed to delete session: {str(e)}"}
    
    def start_service(self):
        def handler_factory(*args, **kwargs):
            return GraspGenHandler(*args, grasp_service=self, **kwargs)
        
        server = HTTPServer((self.host, self.port), handler_factory)
        
        self.logger.info(f"Simple GraspGen service starting on {self.host}:{self.port}")
        self.logger.info("API endpoints:")
        self.logger.info("  POST /create_session")
        self.logger.info("  POST /generate_grasps")
        self.logger.info("  POST /process_pointcloud")
        self.logger.info("  POST /health_check")
        self.logger.info("  POST /delete_session")
        self.logger.info("Press Ctrl+C to stop")
        
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            self.logger.info("Service stopped by user")
        finally:
            server.server_close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple GraspGen HTTP Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8080, help="Port number")
    args = parser.parse_args()
    
    print("=== Simple GraspGen HTTP Server ===")
    print("This is a backup server that doesn't require uvicorn[standard]")
    
    try:
        service = SimpleGraspGenService(host=args.host, port=args.port)
        service.start_service()
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
