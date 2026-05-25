document.addEventListener('DOMContentLoaded', () => {
    // --- Copy Button Logic ---
    const copyBtn = document.getElementById('copyBtn');
    if (copyBtn) {
        copyBtn.addEventListener('click', () => {
            const codeBlock = document.querySelector('.code-wrapper pre code');
            if (codeBlock) {
                navigator.clipboard.writeText(codeBlock.innerText.trim())
                    .then(() => {
                        copyBtn.innerHTML = '<i class="fa-solid fa-check"></i> Copied!';
                        copyBtn.style.color = '#a6e3a1';
                        copyBtn.style.borderColor = 'rgba(166, 227, 161, 0.5)';
                        setTimeout(() => {
                            copyBtn.innerHTML = '<i class="fa-regular fa-copy"></i> Copy';
                            copyBtn.style.color = '#89b4fa';
                            copyBtn.style.borderColor = 'rgba(137, 180, 250, 0.25)';
                        }, 2000);
                    })
                    .catch(err => {
                        console.error('Failed to copy text: ', err);
                    });
            }
        });
    }

    // --- Interactive Physics Simulation ---
    const canvas = document.getElementById('physicsCanvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const toggleWeberBtn = document.getElementById('toggleWeber');
    const resetSimBtn = document.getElementById('resetSim');

    let weberActive = true;
    let particles = [];
    const maxParticles = 40;
    const c_speed = 4.2; // Simulated speed of light constant (c)
    
    // Core target (The local minimum)
    const target = {
        x: 0,
        y: 0,
        radius: 12,
        pulse: 0
    };

    // Resize canvas to container
    function resizeCanvas() {
        const rect = canvas.parentNode.getBoundingClientRect();
        canvas.width = rect.width;
        canvas.height = rect.height;
        target.x = canvas.width / 2;
        target.y = canvas.height / 2;
    }
    
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();

    // Particle class representing parameter coordinates
    class Particle {
        constructor() {
            this.reset();
        }

        reset() {
            // Spawn randomly on outer boundaries
            const angle = Math.random() * Math.PI * 2;
            const dist = 120 + Math.random() * 80;
            this.x = target.x + Math.cos(angle) * dist;
            this.y = target.y + Math.sin(angle) * dist;
            
            // Tangential initial velocity (creating standard orbiting behavior)
            const speed = 1.8 + Math.random() * 1.5;
            this.vx = -Math.sin(angle) * speed;
            this.vy = Math.cos(angle) * speed;

            // Historical states for tracking previous velocity (to compute acceleration)
            this.prev_vx = this.vx;
            this.prev_vy = this.vy;

            this.radius = 3.5 + Math.random() * 2.5;
            // Tail points for drawing motion trails
            this.trail = [];
            this.maxTrail = 12;
            this.color = Math.random() > 0.5 ? '#89b4fa' : '#cba6f7';
        }

        update() {
            // 1. Compute attractive force vector (Standard Gradient Gravity toward local minimum)
            const dx = target.x - this.x;
            const dy = target.y - this.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            
            if (distance < 12) {
                // Particle reached minimum! Reset to keep simulation active
                this.reset();
                return;
            }

            // Normal gravitational pull force magnitude
            const gravityForce = 0.25; 
            const fx = (dx / distance) * gravityForce;
            const fy = (dy / distance) * gravityForce;

            // 2. Apply electrodynamic velocity-acceleration correction (Weber Bracket)
            let weberFactor = 1.0;
            if (weberActive) {
                // Parameter velocity components
                const v_x = this.vx;
                const v_y = this.vy;
                const v_sq = v_x * v_x + v_y * v_y; // Velocity squared

                // Parameter acceleration components (velocity difference)
                const a_x = this.vx - this.prev_vx;
                const a_y = this.vy - this.prev_vy;

                // Relative velocity-acceleration dot product
                const v_dot_a = v_x * a_x + v_y * a_y;

                // W = 1 - v²/(2c²) + (v · a)/c²
                const c_sq = c_speed * c_speed;
                weberFactor = 1.0 - (v_sq / (2.0 * c_sq)) + (v_dot_a / c_sq);
                
                // Keep the correction mathematically bounded to avoid divergence
                weberFactor = Math.max(0.1, Math.min(weberFactor, 2.5));
            }

            // Store current velocity as previous before updating
            this.prev_vx = this.vx;
            this.prev_vy = this.vy;

            // Update velocity scaled by the dynamic Weber factor
            this.vx += fx * weberFactor;
            this.vy += fy * weberFactor;

            // Apply slight physical friction/damping
            const naturalDamping = weberActive ? 0.985 : 0.996;
            this.vx *= naturalDamping;
            this.vy *= naturalDamping;

            // Update position
            this.x += this.vx;
            this.y += this.vy;

            // Motion trail updates
            this.trail.push({x: this.x, y: this.y});
            if (this.trail.length > this.maxTrail) {
                this.trail.shift();
            }
        }

        draw() {
            // Draw motion trail gradient
            if (this.trail.length > 1) {
                ctx.beginPath();
                ctx.moveTo(this.trail[0].x, this.trail[0].y);
                for (let i = 1; i < this.trail.length; i++) {
                    ctx.lineTo(this.trail[i].x, this.trail[i].y);
                }
                ctx.strokeStyle = this.color === '#89b4fa' ? 'rgba(137, 180, 250, 0.15)' : 'rgba(203, 166, 247, 0.15)';
                ctx.lineWidth = this.radius * 0.5;
                ctx.stroke();
            }

            // Draw parameter particle sphere
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
            ctx.fillStyle = this.color;
            ctx.shadowBlur = 10;
            ctx.shadowColor = this.color;
            ctx.fill();
            ctx.shadowBlur = 0; // Reset shadow for canvas performance
        }
    }

    // Spawn initial particle cluster
    for (let i = 0; i < maxParticles; i++) {
        particles.push(new Particle());
    }

    // Main animation loop
    function animate() {
        ctx.fillStyle = '#060709';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw grid lines in the background
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.025)';
        ctx.lineWidth = 1;
        const gridSize = 40;
        for (let x = 0; x < canvas.width; x += gridSize) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, canvas.height);
            ctx.stroke();
        }
        for (let y = 0; y < canvas.height; y += gridSize) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(canvas.width, y);
            ctx.stroke();
        }

        // Draw central local minimum (attractive valley)
        target.pulse += 0.05;
        const targetGlow = target.radius + Math.sin(target.pulse) * 4;
        
        ctx.beginPath();
        ctx.arc(target.x, target.y, targetGlow + 10, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(166, 227, 161, 0.04)';
        ctx.fill();

        ctx.beginPath();
        ctx.arc(target.x, target.y, target.radius, 0, Math.PI * 2);
        ctx.fillStyle = '#a6e3a1';
        ctx.shadowBlur = 20;
        ctx.shadowColor = '#a6e3a1';
        ctx.fill();
        ctx.shadowBlur = 0;

        // Label for the local minimum target
        ctx.font = '700 9px Space Grotesque';
        ctx.fillStyle = 'rgba(166, 227, 161, 0.8)';
        ctx.textAlign = 'center';
        ctx.fillText('LOCAL MINIMUM', target.x, target.y - 20);

        // Update and draw each particle
        particles.forEach(p => {
            p.update();
            p.draw();
        });

        requestAnimationFrame(animate);
    }

    // Start simulation loop
    animate();

    // Toggle button event handler
    if (toggleWeberBtn) {
        toggleWeberBtn.addEventListener('click', () => {
            weberActive = !weberActive;
            if (weberActive) {
                toggleWeberBtn.classList.add('active');
                toggleWeberBtn.innerText = 'Weber Force: ON';
                toggleWeberBtn.style.color = '#a6e3a1';
            } else {
                toggleWeberBtn.classList.remove('active');
                toggleWeberBtn.innerText = 'Weber Force: OFF (Overshooting)';
                toggleWeberBtn.style.color = '#f38ba8';
            }
        });
    }

    // Reset button event handler
    if (resetSimBtn) {
        resetSimBtn.addEventListener('click', () => {
            particles.forEach(p => p.reset());
        });
    }
});
