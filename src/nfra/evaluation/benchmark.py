"""
Benchmarking Suite for NFRA 2.0

Created by Saurav Bhandari
"""

import torch
import time
from typing import Dict, List
from ..models import NFRAForCausalLM, NFRAConfig


class NFRABenchmark:
    """
    Comprehensive benchmarking for NFRA models.
    """
    
    def __init__(self, model: NFRAForCausalLM, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        
    def benchmark_inference(self, seq_lengths: List[int] = [128, 256, 512], 
                           batch_size: int = 4, warmup: int = 3, 
                           repeats: int = 10) -> Dict:
        """Benchmark inference speed across different sequence lengths."""
        results = {}
        
        self.model.eval()
        
        for seq_len in seq_lengths:
            input_ids = torch.randint(
                0, self.model.config.vocab_size, 
                (batch_size, seq_len), 
                device=self.device
            )
            
            # Warmup
            for _ in range(warmup):
                with torch.no_grad():
                    _ = self.model(input_ids)
            
            if self.device == "cuda":
                torch.cuda.synchronize()
            
            # Benchmark
            start = time.time()
            for _ in range(repeats):
                with torch.no_grad():
                    _ = self.model(input_ids)
            
            if self.device == "cuda":
                torch.cuda.synchronize()
            
            elapsed = time.time() - start
            tokens_per_sec = (batch_size * seq_len * repeats) / elapsed
            
            results[seq_len] = {
                "tokens_per_second": round(tokens_per_sec, 2),
                "ms_per_token": round(1000 / tokens_per_sec, 3)
            }
            
        return results
    
    def benchmark_energy(self, energy_budgets: List[float] = [0.3, 0.5, 0.7, 1.0],
                        seq_len: int = 256, batch_size: int = 4) -> Dict:
        """Test performance under different energy constraints."""
        results = {}
        input_ids = torch.randint(
            0, self.model.config.vocab_size,
            (batch_size, seq_len),
            device=self.device
        )
        
        for budget in energy_budgets:
            self.model.eval()
            with torch.no_grad():
                start = time.time()
                outputs = self.model(input_ids, energy_budget=budget)
                elapsed = time.time() - start
                
            results[budget] = {
                "time_seconds": round(elapsed, 4),
                "tokens_per_second": round((batch_size * seq_len) / elapsed, 2)
            }
            
        return results
    
    def run_full_benchmark(self) -> Dict:
        """Run complete benchmark suite."""
        print("Running full NFRA benchmark...")
        
        inference_results = self.benchmark_inference()
        energy_results = self.benchmark_energy()
        sparsity = self._get_model_sparsity()
        
        return {
            "inference": inference_results,
            "energy_efficiency": energy_results,
            "sparsity": sparsity,
            "device": self.device
        }
    
    def _get_model_sparsity(self):
        from .metrics import compute_sparsity
        return compute_sparsity(self.model)