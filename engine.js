/**
 * Scalar Value object, used for automatic differentiation (autograd)
 */
export class Value {
    // Defining private properties 
    #prev;
    #backward;

    /**
     * 
     * @param {number} data - raw numerical scalar 
     * @param {Value[]} [prev=[]] - Parent nodes in the computational graph 
     */
	constructor (data, prev = []) {
		this.data = data;
		this.grad = 0;
		this.#prev = new Set(prev);
		this.#backward = () => {};
	}

    /** Helper to guarantee a Value object */
    static from(val) {
        return val instanceof Value ? val : new Value(val);
    }


    // Arithmetic operations

	add (other) {
		other = Value.from(other);

		const out = new Value(this.data + other.data, [this, other]);

		out.#backward = () => {
			this.grad += 1 * out.grad;
			other.grad += 1 * out.grad;
		};

		return out;
	}


	mult (other) {
		other = Value.from(other);

		const out = new Value (this.data * other.data, [this, other]);

		out.#backward = () => {
			this.grad += other.data * out.grad;
			other.grad += this.data * out.grad;
		};

		return out;
	}

	// other here is not a value object but a number for my uses.
	pow (exponent) {
        if (typeof exponent !== 'number') {
            throw new TypeError(`pow() expects a number as the exponent, got ${typeof exponent}`);
        }

		const out = new Value(this.data ** exponent, [this]);

		out.#backward = () => {
			this.grad  += exponent * (this.data ** (exponent - 1)) * out.grad;
		}

		return out;
	}

    div (other) {
        return this.mult(Value.from(other).pow(-1));
    }

	neg () {
		return this.mult(-1);
	}

	sub (other) {
		return this.add(Value.from(other).neg());
	}

    // Activation functions

	tanh () {
		const t = Math.tanh(this.data);
		const out = new Value(t, [this]);

		out.#backward = () => {
			this.grad += (1 - t ** 2) * out.grad;
		}
		return out;
	}

    relu () {
        const out = new Value(this.data < 0 ? 0 : this.data, [this]);

        out.#backward = () => {
            this.grad += (this.data < 0 ? 0 : 1) * out.grad;
        };

        return out;

    }

    exp () {
        const x = Math.exp(this.data);
        const out = new Value(x, [this]);

        out.#backward = () => {
            this.grad += x * out.grad;
        };

        return out;
    }

    // Autograd Graph Traversal and Backpropagation
	backward () {
		const topo = [];
        const visited  = new Set();

		const build_topo = (node) => {
			if (!visited.has(node)) {
				visited.add(node)
				for(const child of node.#prev) {
					build_topo(child)
				}
                topo.push(node)
			}
		};

		build_topo(this);

        // reset seed gradient to 1 (dL/dL = 1)
		this.grad = 1.0;

		for (const node of topo.reverse()) {
			node.#backward();
		}
	}

    // Utility Methods
    /** Coerces Value object to a primitive Number for math expressions/comparisons */
    valueOf () {
        return this.data;
    }

    /** Formats Object into a clean string for logging */
    toString () {
        return `Value(data=${this.data.toFixed(4)}, grad=${this.grad.toFixed(4)})`;
    }

}

