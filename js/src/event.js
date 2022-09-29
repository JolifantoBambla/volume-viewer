class UIEvent extends Event {
    detail;
    type;

    constructor(type, options) {
        super(type, options);
        this.detail = options.detail;
        this.type = type;
    }
}

class MouseEvent extends UIEvent {
    altKey;
    button;
    buttons;
    clientX;
    clientY;
    ctrlKey;
    layerX;
    layerY;
    metaKey;
    movementX;
    movementY;
    offsetX;
    offsetY;
    pageX;
    pageY;
    relatedTarget;
    screenX;
    screenY;
    mozInputSource;
    webkitForce;
    x;
    y;

    constructor(type, options) {
        super(type, options);
        this.altKey = options.altKey;
        this.button = options.button;
        this.buttons = options.buttons;
        this.clientX = options.clientX;
        this.clientY = options.clientY;
        this.ctrlKey = options.ctrlKey;
        this.layerX = options.layerX;
        this.layerY = options.layerY;
        this.metaKey = options.metaKey;
        this.movementX = options.movementX;
        this.movementY = options.movementY;
        this.offsetX = options.offsetX;
        this.offsetY = options.offsetY;
        this.pageX = options.pageX;
        this.pageY = options.pageY;
        this.relatedTarget = options.relatedTarget;
        this.screenX = options.screenX;
        this.screenY = options.screenY;
        this.mozInputSource = options.mozInputSource;
        this.webkitForce = options.webkitForce;
        this.x = options.x;
        this.y = options.y;
    }

    static fromRealMouseEvent(event) {
        return new MouseEvent(
            event.type,
            {
                altKey: event.altKey,
                button: event.button,
                buttons: event.buttons,
                clientX: event.clientX,
                clientY: event.clientY,
                ctrlKey: event.ctrlKey,
                layerX: event.layerX,
                layerY: event.layerY,
                metaKey: event.metaKey,
                movementX: event.movementX,
                movementY: event.movementY,
                offsetX: event.offsetX,
                offsetY: event.offsetY,
                pageX: event.pageX,
                pageY: event.pageY,
                relatedTarget: event.relatedTarget,
                screenX: event.screenX,
                screenY: event.screenY,
                mozInputSource: event.mozInputSource,
                webkitForce: event.webkitForce,
                x: event.x,
                y: event.y,
            });
    }
}

class WheelEvent extends MouseEvent {
    deltaMode;
    deltaX;
    deltaY;
    deltaZ;

    constructor(type, options) {
        super(type, options);
        this.deltaMode = options.deltaMode;
        this.deltaX = options.deltaX;
        this.deltaY = options.deltaY;
        this.deltaZ = options.deltaZ;
    }

    static fromRealWheelEvent(event) {
        return new WheelEvent(
            event.type,
            {
                deltaMode: event.deltaMode,
                deltaX: event.deltaX,
                deltaY: event.deltaY,
                deltaZ: event.deltaZ,
            });
    }
}

class KeyboadEvent extends UIEvent {
    altKey;
    code;
    ctrlKey;
    isComposing;
    key;
    locale;
    location;
    metaKey;
    repeat;
    shiftKey;

    constructor(type, options) {
        super(type, options);
        this.altKey = options.altKey;
        this.code = options.code;
        this.ctrlKey = options.ctrlKey;
        this.isComposing = options.isComposing;
        this.key = options.key;
        this.locale = options.locale;
        this.location = options.location;
        this.metaKey = options.metaKey;
        this.repeat = options.repeat;
        this.shiftKey = options.shiftKey;
    }

    static fromRealKeyboardEvent(event) {
        return new KeyboadEvent(
            event.type,
            {
                altKey: event.altKey,
                code: event.code,
                ctrlKey: event.ctrlKey,
                isComposing: event.isComposing,
                key: event.key,
                locale: event.locale,
                location: event.location,
                metaKey: event.metaKey,
                repeat: event.repeat,
                shiftKey: event.shiftKey,
            });
    }
}

function toWrappedEvent(event) {
    switch (event.type) {
        case 'click':
        case 'dblclick':
        case 'mousedown':
        case 'mouseup':
        case 'mousemove':
        case 'mouseover':
        case 'mouseout':
        case 'mouseenter':
        case 'mouseleave':
            return MouseEvent.fromRealMouseEvent(event);
        case 'wheel':
            return WheelEvent.fromRealWheelEvent(event);
        case 'keydown':
        case 'keyup':
        case 'keypress':
            return KeyboadEvent.fromRealKeyboardEvent(event);
        default:
            return event;
    }
}

export {
    toWrappedEvent,
};
