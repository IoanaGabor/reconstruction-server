from typing import List
from reconstruction_service.simple_reconstruction_pipeline import BrainScanReconstructionPipeline
import io
import numpy as np
import torch
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image

class ReconstructionService:
    def reconstruct_image(self, person_id: int, voxels: List[int]):
        if not voxels:
            raise ValueError("Voxel array cannot be empty")
        
        pipe = BrainScanReconstructionPipeline();
        image = pipe.run(voxels, person_id)

        # Save image to an in-memory buffer
        img_io = io.BytesIO()
        image.save(img_io, format="PNG")
        img_io.seek(0)

        # Return image as response
        return StreamingResponse(img_io, media_type="image/png")

        #return {"person_id": person_id, "image": image}
