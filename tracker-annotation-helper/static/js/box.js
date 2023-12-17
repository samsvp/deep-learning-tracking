class Square {
    static getHandleSize() {
        return 10;
    }

    // get a random color
    static getRandomColor(seed) {
        let letters = '0123456789ABCDEF';
        let color = '#';

        for (let i = 0; i < 6; i++) {
            let r = Math.floor((Math.abs(Math.sin((seed + i)) * 43758.5453123) % 1) * 16);
            color += letters[r];
        }
        return color;
    }

    constructor(x, y, width, height, number) {
        this.x = x;
        this.y = y;
        this.lastX = x;
        this.lastY = y;
        this.width = width;
        this.height = height;
        this.handleSize = Square.getHandleSize();
        this.number = number === undefined ? -1 : parseInt(number);
        this.color = Square.getRandomColor(this.number);

        this.isDragging = false;
        this.isResizing = false;
        this.mMouseOver = false;
        this.resizeHandle = null;
        this.lastX = 0;
        this.lastY = 0;
    }
    

    setNumber(number) {
        this.number = parseInt(number);
        this.color = Square.getRandomColor(this.number);
    }


    draw(canvas, ctx, img, clear = false) {
        if (clear) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0);
        }
        ctx.lineWidth = 2;

        let alpha = this.mMouseOver ? "AA" : "33";
        ctx.fillStyle = this.color + alpha;
        ctx.fillRect(this.x, this.y, this.width, this.height);
        ctx.strokeStyle = this.color;
        ctx.strokeRect(this.x, this.y, this.width, this.height);

        // Draw the resize handle
        let handleX = this.x + this.width - this.handleSize;
        let handleY = this.y + this.height - this.handleSize;
        ctx.fillStyle = 'white';
        ctx.fillRect(handleX, handleY, this.handleSize, this.handleSize);
        ctx.strokeStyle = 'black';
        ctx.strokeRect(handleX, handleY, this.handleSize, this.handleSize);

        // Draw the text
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.strokeStyle = 'black';
        ctx.strokeText(this.number, this.x + (this.width / 2), this.y - 5);
        ctx.fillStyle = this.color;
        ctx.fillText(this.number, this.x + (this.width / 2), this.y - 5);
    }


    update(mouseX, mouseY) {
        let shouldRedraw = false;
        this.mMouseOver = this.isMouseOver(mouseX, mouseY);

        if (this.isResizing) {
            let deltaX = mouseX - this.resizeHandle.x;
            let deltaY = mouseY - this.resizeHandle.y;
            this.width = this.resizeHandle.width + deltaX;
            this.height = this.resizeHandle.height + deltaY;
            shouldRedraw = true;
        }

        if (this.isDragging) {
            let deltaX = mouseX - this.lastX;
            let deltaY = mouseY - this.lastY;
            this.x += deltaX;
            this.y += deltaY;

            shouldRedraw = true;
        } 

        this.lastX = mouseX;
        this.lastY = mouseY;

        return shouldRedraw;
    }


    setAsResizing(mouseX, mouseY) {
        this.isResizing = true;
        this.resizeHandle = {
            x: mouseX,
            y: mouseY,
            width: this.width,
            height: this.height
        }
    }


    // Check if the mouse is over the resize handle
    isOverResizeHandle(x, y) {
        let handleX = this.x + this.width - this.handleSize;
        let handleY = this.y + this.height - this.handleSize;
        return x >= handleX && x <= handleX + this.handleSize &&
            y >= handleY && y <= handleY + this.handleSize;
    }


    isMouseOver(mouseX, mouseY) {
        return (mouseX >= this.x && mouseX <= this.x + this.width &&
            mouseY >= this.y && mouseY <= this.y + this.height);
    }
}