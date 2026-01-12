import asyncio
from asyncio import sleep

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellTimeoutError, DeadKernelError
from nbformat.v4 import new_code_cell
from rich.console import Console

nb = nbformat.v4.new_notebook()
nb_client = NotebookClient(nb, timeout=600)
console = Console()


async def run(cell):
    if nb_client.kc is None or not await nb_client.kc.is_alive():
        nb_client.create_kernel_manager()
        nb_client.start_new_kernel()
        nb_client.start_new_kernel_client()
    cell_index = nb.cells.index(cell)
    # note: 历史代码块及其执行结果
    for cell in nb.cells:
        Console().print(cell.outputs)
    try:
        await nb_client.async_execute_cell(cell, cell_index)
        # note: 当前执行结果
        # console.print(nb.cells[-1].outputs)
    except CellTimeoutError:
        console.print("Cell execution timed out.")
    except DeadKernelError:
        console.print("Kernel died during execution.")


async def main():
    nb.cells.append(new_code_cell(source="print('hello world')"))
    print("-----------ready to run 1---------------")
    await run(nb.cells[-1])
    nb.cells.append(new_code_cell(source="print('hello world2')"))
    print("-----------ready to run 2---------------")
    await sleep(10)
    await run(nb.cells[-1])
    nb.cells.append(new_code_cell(source="print('hello world3')"))
    print("-----------ready to run 3---------------")
    await sleep(10)
    await run(nb.cells[-1])


asyncio.run(main())
