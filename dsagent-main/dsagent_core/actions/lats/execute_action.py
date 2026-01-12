"""Execute action for LATS using MetaGPT's ExecuteNbCode."""

from typing import Tuple, Any
from metagpt.actions import ExecuteNbCode
from metagpt.logs import logger


class ExecuteAction:
    """
    Execute action wrapper for LATS.
    Adapts MetaGPT's ExecuteNbCode for LATS tree search.
    """
    
    def __init__(self):
        self.execute_nb_code = ExecuteNbCode()
    
    async def step(
        self, 
        goal: str, 
        instruction: str, 
        code: str, 
        execute_code: ExecuteNbCode
    ) -> Tuple[str, float, bool, dict]:
        """
        Execute code and return observation, reward, success status, and info.
        
        Args:
            goal: The overall goal/task
            instruction: Human-readable instruction
            code: Code to execute
            execute_code: ExecuteNbCode instance to use
            
        Returns:
            Tuple of (observation, reward, is_success, info)
        """
        try:
            # Use the provided execute_code instance
            result = await execute_code.run(code)
            
            # Extract observation from result
            observation = str(result.get("output", "")) if isinstance(result, dict) else str(result)
            
            # Determine success (simplified heuristic)
            is_success = "error" not in observation.lower() and "traceback" not in observation.lower()
            
            # Calculate reward (0-10 scale)
            if not is_success:
                reward = 0.0
            elif not observation or observation.strip() == "":
                reward = 3.0  # Executed but no output
            else:
                reward = 7.0  # Successfully executed with output
            
            # Build info dict
            info = {
                "goal": goal,
                "instruction": instruction,
                "code": code,
                "result": result
            }
            
            return observation, reward, is_success, info
            
        except Exception as e:
            logger.error(f"ExecuteAction failed: {e}")
            observation = f"Error: {str(e)}"
            return observation, 0.0, False, {"error": str(e)}
