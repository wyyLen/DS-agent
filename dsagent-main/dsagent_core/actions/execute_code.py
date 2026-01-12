"""
Independent Code Executor - No dependency on MetaGPT.

This module provides a standalone code execution environment using Jupyter kernels.
"""

import asyncio
import base64
import re
import logging
from typing import Tuple, Literal, Optional
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionComplete, CellTimeoutError, DeadKernelError
from nbclient.util import ensure_async
from nbformat import NotebookNode
from nbformat.v4 import new_code_cell, new_output, output_from_msg

logger = logging.getLogger(__name__)

# Initialization code to suppress warnings
INI_CODE = """import warnings
import logging

root_logger = logging.getLogger()
root_logger.setLevel(logging.ERROR)
warnings.filterwarnings('ignore')"""


def remove_escape_and_color_codes(text: str) -> str:
    """Remove ANSI escape codes and color codes from text."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


def remove_log_and_warning_lines(text: str) -> str:
    """Remove log and warning lines from output."""
    lines = text.split('\n')
    filtered_lines = []
    for line in lines:
        if not any(tag in line for tag in ['WARNING', 'DEBUG', 'INFO', 'ERROR']):
            filtered_lines.append(line)
    return '\n'.join(filtered_lines)


class SimpleOutputNotebookClient(NotebookClient):
    """Notebook client with simplified output handling."""

    def __init__(self, *args, output_callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_callback = output_callback

    async def _async_poll_output_msg(
        self, 
        parent_msg_id: str, 
        cell: NotebookNode, 
        cell_index: int
    ) -> None:
        """Poll for output messages."""
        assert self.kc is not None
        while True:
            msg = await ensure_async(self.kc.iopub_channel.get_msg(timeout=None))
            
            # Call output callback if provided
            if self.output_callback:
                await self.output_callback(msg)

            if msg["parent_header"].get("msg_id") == parent_msg_id:
                try:
                    self.process_message(msg, cell, cell_index)
                except CellExecutionComplete:
                    return


class IndependentCodeExecutor:
    """
    Independent code executor using Jupyter kernel.
    
    No dependencies on MetaGPT - can be used standalone.
    """
    
    def __init__(
        self,
        timeout: int = 600,
        workspace_path: Optional[Path] = None
    ):
        """
        Initialize code executor.
        
        Args:
            timeout: Execution timeout in seconds
            workspace_path: Working directory for code execution
        """
        self.timeout = timeout
        self.workspace_path = workspace_path or Path.cwd()
        self.nb = nbformat.v4.new_notebook()
        self.nb_client: Optional[SimpleOutputNotebookClient] = None
        self.init_called = False
        
        logger.info(f"IndependentCodeExecutor initialized with workspace: {self.workspace_path}")
    
    async def _ensure_initialized(self):
        """Ensure kernel is initialized and ready."""
        if self.nb_client is None:
            await self._create_nb_client()
        
        if not self.init_called:
            await self._init_kernel()
    
    async def _create_nb_client(self):
        """Create notebook client."""
        self.nb_client = SimpleOutputNotebookClient(
            self.nb,
            timeout=self.timeout,
            resources={"metadata": {"path": str(self.workspace_path)}}
        )
        
        # Start kernel
        if self.nb_client.kc is None or not await self.nb_client.kc.is_alive():
            self.nb_client.create_kernel_manager()
            self.nb_client.start_new_kernel()
            self.nb_client.start_new_kernel_client()
            logger.info("Jupyter kernel started")
    
    async def _init_kernel(self):
        """Initialize kernel with suppression code."""
        if not self.init_called:
            # Execute initialization code directly without calling run()
            # to avoid recursion
            cell = new_code_cell(source=INI_CODE)
            self.nb.cells.append(cell)
            cell_index = len(self.nb.cells) - 1
            
            try:
                await self.nb_client.async_execute_cell(cell, cell_index)
            except Exception as e:
                logger.warning(f"Failed to initialize kernel: {e}")
            
            self.init_called = True
            logger.info("Kernel initialized with warning suppression")
    
    async def run(self, code: str) -> Tuple[str, bool]:
        """
        Execute Python code and return results.
        
        Args:
            code: Python code to execute
            
        Returns:
            Tuple of (output_text, success_flag)
        """
        await self._ensure_initialized()
        
        # Create code cell
        cell = new_code_cell(source=code)
        self.nb.cells.append(cell)
        cell_index = len(self.nb.cells) - 1
        
        try:
            # Execute cell
            await self.nb_client.async_execute_cell(cell, cell_index)
            return self._parse_outputs(self.nb.cells[-1].outputs)
            
        except CellTimeoutError:
            # Interrupt kernel on timeout
            if self.nb_client.km is not None:
                await self.nb_client.km.interrupt_kernel()
                await asyncio.sleep(1)
            error_msg = (
                "Cell execution timed out: Execution exceeded the time limit and was stopped; "
                "consider optimizing your code for better performance."
            )
            logger.warning(error_msg)
            return error_msg, False
            
        except DeadKernelError:
            # Reset kernel on death
            logger.error("Kernel died, resetting...")
            await self.reset()
            return "DeadKernelError: Kernel died and was reset", False
            
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return self._parse_outputs(self.nb.cells[-1].outputs)
    
    def _parse_outputs(
        self, 
        outputs: list, 
        keep_len: int = 5000
    ) -> Tuple[str, bool]:
        """
        Parse outputs from cell execution.
        
        Args:
            outputs: List of output objects
            keep_len: Maximum length of output to keep
            
        Returns:
            Tuple of (parsed_output, success_flag)
        """
        if not outputs:
            return "", True
        
        parsed_output = []
        is_success = True
        
        for output in outputs:
            output_text = ""
            
            if output["output_type"] == "stream":
                output_text = output.get("text", "")
                
            elif output["output_type"] == "display_data":
                # Handle image data
                if "image/png" in output.get("data", {}):
                    output_text = "[Image output - base64 data omitted]"
                elif "text/plain" in output.get("data", {}):
                    output_text = output["data"]["text/plain"]
                    
            elif output["output_type"] == "execute_result":
                output_text = output.get("data", {}).get("text/plain", "")
                
            elif output["output_type"] == "error":
                output_text = "\n".join(output.get("traceback", []))
                is_success = False
            
            # Clean output
            output_text = remove_escape_and_color_codes(output_text)
            
            if is_success:
                output_text = remove_log_and_warning_lines(output_text)
            
            # Truncate long outputs
            if "<!DOCTYPE html>" not in output_text:
                if is_success:
                    output_text = output_text[:keep_len]
                else:
                    # For errors, keep the end (most relevant)
                    output_text = output_text[-keep_len:]
            
            if output_text.strip():
                parsed_output.append(output_text)
        
        return "\n".join(parsed_output), is_success
    
    async def reset(self):
        """Reset the execution environment."""
        logger.info("Resetting code executor...")
        await self.terminate()
        await asyncio.sleep(1)
        await self._create_nb_client()
        self.init_called = False
        await self._ensure_initialized()
        logger.info("Code executor reset complete")
    
    async def terminate(self):
        """Terminate the kernel and cleanup resources."""
        if self.nb_client is None:
            return
        
        if self.nb_client.km is not None and await self.nb_client.km.is_alive():
            await self.nb_client.km.shutdown_kernel(now=True)
            await self.nb_client.km.cleanup_resources()
            
            # Stop all channels
            channels = [
                self.nb_client.kc.stdin_channel,
                self.nb_client.kc.hb_channel,
                self.nb_client.kc.control_channel,
            ]
            
            for channel in channels:
                if channel.is_alive():
                    channel.stop()
            
            self.nb_client.kc = None
            self.nb_client.km = None
        
        logger.info("Kernel terminated")
    
    def clear_outputs(self):
        """Clear all output from cells."""
        for cell in self.nb.cells:
            if "outputs" in cell:
                cell["outputs"] = []
    
    def get_notebook(self) -> NotebookNode:
        """Get the current notebook object."""
        return self.nb
    
    def save_notebook(self, path: Path):
        """Save notebook to file."""
        with open(path, 'w', encoding='utf-8') as f:
            nbformat.write(self.nb, f)
        logger.info(f"Notebook saved to {path}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_initialized()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.terminate()


# Convenience function
async def execute_code(
    code: str,
    timeout: int = 600,
    workspace_path: Optional[Path] = None
) -> Tuple[str, bool]:
    """
    Execute code in a fresh environment.
    
    Args:
        code: Python code to execute
        timeout: Execution timeout
        workspace_path: Working directory
        
    Returns:
        Tuple of (output, success)
    """
    async with IndependentCodeExecutor(timeout, workspace_path) as executor:
        return await executor.run(code)
