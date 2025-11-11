#!/usr/bin/env python3
"""
Simplified GraspGen client - can run outside Docker

This file only contains client code, does not depend on GraspGen modules,
and can run on the host machine or other machines.
"""

import numpy as np
from pathlib import Path
import pickle
import requests
import time
from typing import Dict, Any, Tuple


class GraspGenInterface:
    """Simplified GraspGen client that can run outside Docker"""
    
    def __init__(self, 
                 gripper_config: str,
                 host: str = "localhost", port: int = 7000, ):
        """
        Initialize client

        Args:
            host: Server address (Docker host IP)
            port: Server port
            gripper_config: Gripper configuration file path (path inside container)
        """
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.session_id = None

        # Create session
        self._create_session(gripper_config)

    def _create_session(self, gripper_config: str):
        """Create session"""
        params = {"gripper_config": gripper_config}
        
        response = requests.post(
            f"{self.base_url}/create_session",
            data=pickle.dumps(params),
            headers={'Content-Type': 'application/octet-stream'},
            timeout=30
        )
        response.raise_for_status()
        
        result = pickle.loads(response.content)
        if "error" in result:
            raise RuntimeError(f"Failed to create session: {result['error']}")
        
        self.session_id = result["session_id"]
        print(f"Connected to GraspGen service, session: {self.session_id}")
    
    def generate_grasps(self, point_cloud: np.ndarray,
                       grasp_threshold: float = 0.8,
                       num_grasps: int = 200,
                       topk_num_grasps: int = -1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate grasp poses

        Args:
            point_cloud: Point cloud data (N, 3)
            grasp_threshold: Grasp threshold
            num_grasps: Number of grasps to generate
            topk_num_grasps: Return top K grasps

        Returns:
            (grasps, confidences): Grasp poses and confidence scores
        """
        if self.session_id is None:
            raise RuntimeError("No active session")
        
        data = {
            "session_id": self.session_id,
            "point_cloud": point_cloud,
            "grasp_threshold": grasp_threshold,
            "num_grasps": num_grasps,
            "topk_num_grasps": topk_num_grasps,
        }
        
        response = requests.post(
            f"{self.base_url}/generate_grasps",
            data=pickle.dumps(data),
            headers={'Content-Type': 'application/octet-stream'},
            timeout=120
        )
        response.raise_for_status()
        
        result = pickle.loads(response.content)
        if "error" in result:
            raise RuntimeError(f"Failed to generate grasps: {result['error']}")
        
        print(f"✓ Generated {result['num_grasps']} grasps in {result['inference_time']:.3f}s")
        return result["grasps"], result["grasp_confidence"]
    
    def health_check(self) -> Dict[str, Any]:
        """Check service health status"""
        response = requests.post(
            f"{self.base_url}/health_check",
            data=pickle.dumps({}),
            headers={'Content-Type': 'application/octet-stream'},
            timeout=10
        )
        response.raise_for_status()
        return pickle.loads(response.content)
    
    def close(self):
        """Close session"""
        if self.session_id is None:
            return
        
        data = {"session_id": self.session_id}
        try:
            response = requests.post(
                f"{self.base_url}/delete_session",
                data=pickle.dumps(data),
                headers={'Content-Type': 'application/octet-stream'},
                timeout=30
            )
            response.raise_for_status()
            print(f"Session {self.session_id} closed")
        except:
            pass
        finally:
            self.session_id = None
