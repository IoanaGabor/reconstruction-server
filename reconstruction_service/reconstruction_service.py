from typing import List
from reconstruction_service.simple_reconstruction_pipeline import BrainScanReconstructionPipeline

class ReconstructionService:
    def reconstruct_image(self, person_id: int, voxels: List[int]):
        if not voxels:
            raise ValueError("Voxel array cannot be empty")
        
        pipe = BrainScanReconstructionPipeline();
        pipe.run(voxels, person_id)

        return {"person_id": person_id, "image_url": f"/static/reconstruction_{person_id}.png"}
